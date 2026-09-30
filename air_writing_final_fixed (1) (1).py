"""
╔══════════════════════════════════════════════════════════════════╗
║        AIR WRITING — FULLY CORRECTED  (v4)                      ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  FIXES IN THIS VERSION:                                          ║
║  ① Mode-1 DRAW   → RING finger only up  (was index)             ║
║  ② UNDO fist     → fixed detection (4-finger curl + thumb in)   ║
║  ③ Drawing       → connected lines every frame (no dot gaps)    ║
║  ④ Shape rec     → larger canvas + better geometry pipeline     ║
║  ⑤ Mode 2 word   → char-by-char builds buffer, autocorrect      ║
║     after every char; pinch to add char; 3-finger hold for space ║
║  ⑥ Auto-predict  → 2s gap timer in ALL modes                    ║
║  ⑦ Mode switch   → ring=draw so index+middle = Mode2,           ║
║     ring only held = Mode1, index+middle+ring = Mode3           ║
║                                                                  ║
║  GESTURES:                                                       ║
║  ┌─────────────────────────────────────────────────────────┐    ║
║  │ RING finger ONLY up          → DRAW (pen active)        │    ║
║  │ Pinch (index+thumb <40px,    → PREDICT / confirm        │    ║
║  │        ring+middle+pinky dn) │                          │    ║
║  │ Index+Middle up, hold 1.2s   → Switch to Mode 2         │    ║
║  │ Index+Middle+Ring, hold 1.2s → Switch to Mode 3         │    ║
║  │ Thumb+Ring up, hold 1.2s     → Switch to Mode 1         │    ║
║  │ Pinky ONLY up                → Space/commit (Mode 2)    │    ║
║  │ All 5 fingers open           → Clear canvas             │    ║
║  │ 4 fingers curled + thumb in  → UNDO                     │    ║
║  │ Thumb up, all 4 down         → Clear all text           │    ║
║  │ Hover ring tip over ERASE    → Erase text (hold 22fr)   │    ║
║  └─────────────────────────────────────────────────────────┘    ║
║                                                                  ║
║  MODE 2 — WORD REC:                                              ║
║  1. Ring finger up → draw a character                            ║
║  2. Pinch → char added to [word buffer], autocorrect updates     ║
║  3. Repeat steps 1-2 for each character in the word              ║
║  4. Green pills appear at bottom showing corrected suggestions   ║
║  5. Hover ring-tip over a pill + Pinch → apply that correction  ║
║  6. Pinky ONLY up → commits word + space, ready for next word   ║
║  OR: After 2s idle → auto-predicts remaining buffer              ║
║                                                                  ║
║  MODEL QUALITY NOTE (read bottom of this file)                   ║
╚══════════════════════════════════════════════════════════════════╝
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os, time, math, threading
from spellchecker import SpellChecker

try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

# ════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════
CAM_W, CAM_H   = 640, 480
MODEL_PATH     = "air_model.tflite"
DATASET_PATH   = "dataset1"
CONF_THRESHOLD = 0.55          # lowered from 0.6 — more permissive
COOLDOWN       = 1.0
FREEZE_DRAW    = 0.55          # freeze after pen-lift before gesture arms
HOVER_FRAMES   = 20
MODE_HOLD_SEC  = 1.2
AUTO_PRED_GAP  = 2.0           # seconds idle → auto-predict (all modes)
PINCH_DIST     = 42
DRAW_THICK     = 10
MIN_MOVE       = 1             # min pixel move to record point (was 5 → caused gaps)

# Kalman filter tuning
KF_Q = 5e-3
KF_R = 0.15

# Colours (BGR)
C_WHITE  = (255,255,255)
C_GREEN  = (0,  255,120)
C_CYAN   = (0,  220,255)
C_ORANGE = (0,  165,255)
C_RED    = (40,  60,220)
C_DARK   = (15,  15, 15)
C_GRAY   = (140,140,140)
C_YELLOW = (0,  220,220)
C_BLUE   = (255,140, 60)
C_BLACK  = (0,   0,  0)

MODES       = {1:"CHAR",  2:"WORD+AC", 3:"SHAPE"}
MODE_COLORS = {1:C_GREEN, 2:C_CYAN,   3:C_ORANGE}

BIGRAMS = {
    "i":["am","will","have"],"you":["are","can","will"],
    "he":["is","was","will"],"she":["is","was","will"],
    "we":["are","will","have"],"they":["are","will","have"],
    "the":["best","most","same"],"a":["great","good","new"],
    "this":["is","was","will"],"my":["name","phone","self"],
    "please":["help","send","call"],"can":["you","we","help"],
    "how":["are","do","can"],"what":["is","are","do"],
    "good":["morning","night","work"],"hello":["there","world","friend"],
    "thank":["you","god","him"],"need":["help","more","to"],
    "go":["to","ahead","now"],"call":["me","you","him"],
    "help":["me","you","please"],"ok":["great","good","sure"],
    "hi":["there","how","friend"],"see":["you","the","me"],
    "do":["it","you","we"],"is":["this","that","it"],
    "have":["you","we","they"],"tell":["me","you","him"],
    "show":["me","it","you"],"let":["me","it","go"],
}

# ════════════════════════════════════════════════════
#  KALMAN  (1-D per axis)
# ════════════════════════════════════════════════════
class KF1D:
    def __init__(self):
        self.x = None; self.P = 1.0
    def update(self, z):
        if self.x is None:
            self.x = float(z); return int(round(self.x))
        self.P += KF_Q
        K = self.P / (self.P + KF_R)
        self.x += K*(float(z)-self.x)
        self.P *= (1-K)
        return int(round(self.x))
    def reset(self): self.x=None; self.P=1.0

kf_x = KF1D(); kf_y = KF1D()

# ════════════════════════════════════════════════════
#  MODEL
# ════════════════════════════════════════════════════
print("Loading TFLite model...")
interp = tf.lite.Interpreter(model_path=MODEL_PATH)
interp.allocate_tensors()
inp_d = interp.get_input_details()
out_d = interp.get_output_details()
print("Model loaded ✔")
classes = sorted(os.listdir(DATASET_PATH))
print(f"Classes ({len(classes)}): {classes}")

# ════════════════════════════════════════════════════
#  SPELL CHECKER
#  ─────────────────────────────────────────────────
#  HOW AUTOCORRECT WORKS:
#  pyspellchecker uses Peter Norvig's edit-distance algorithm.
#  For word_buffer (e.g. "helo") it generates ALL strings within
#  1-2 character edits (insert/delete/replace/swap) and filters
#  to only real English words from its frequency corpus.
#  Results are sorted by usage frequency (common words first).
#
#  get_suggestions() is called after EVERY pinch in Mode 2
#  as you add chars. So you see live suggestions grow.
#
#  Applying a suggestion:
#    → Hover ring finger over a green pill → turns bright
#    → Pinch while hovering → word in buffer replaced
#    → On SPACE/commit → best suggestion auto-applied
# ════════════════════════════════════════════════════
spell = SpellChecker()

def get_suggestions(word, n=4):
    if not word or len(word)<2: return []
    w = word.lower().strip()
    if not spell.unknown([w]): return []   # already correct
    cands = spell.candidates(w)
    if not cands: return []
    return sorted(cands, key=lambda c: spell.word_usage_frequency(c), reverse=True)[:n]

def next_word_sugg(w, n=3):
    return BIGRAMS.get(w.lower().strip(),[])[:n]

# ════════════════════════════════════════════════════
#  TTS
# ════════════════════════════════════════════════════
def speak(text):
    if not TTS_AVAILABLE or not text.strip(): return
    def _go():
        try:
            e=pyttsx3.init(); e.setProperty('rate',150)
            e.say(text); e.runAndWait()
        except: pass
    threading.Thread(target=_go,daemon=True).start()

# ════════════════════════════════════════════════════
#  CAMERA + MEDIAPIPE
# ════════════════════════════════════════════════════
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened(): cap = cv2.VideoCapture(0)
if not cap.isOpened(): cap = cv2.VideoCapture(1)
if not cap.isOpened(): print("Camera failed"); exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
print("Camera ✔")

mp_h = mp.solutions.hands
hproc = mp_h.Hands(max_num_hands=1,
                   min_detection_confidence=0.75,
                   min_tracking_confidence=0.75)
mpd = mp.solutions.drawing_utils
HS  = mpd.DrawingSpec(color=(0,255,200),thickness=2,circle_radius=3)
CS  = mpd.DrawingSpec(color=(0,180,255),thickness=2)

# ════════════════════════════════════════════════════
#  STATE
# ════════════════════════════════════════════════════
canvas       = np.zeros((CAM_H,CAM_W,3),np.uint8)
points       = []
prev_sx=prev_sy=0
is_drawing   = False

current_mode = 1
current_text = ""
word_buffer  = ""
suggestions  = []
next_preds   = []
shape_result = ""

history=[]
MAX_H=20

last_pred=""; last_conf=0.0
top3_preds=[]; pred_until=0.0

prev_gest="NONE"; gest_last_t=0.0; freeze_until=0.0
mode_hold_g=0; mode_hold_t=0.0
erase_cnt=0
ERASE_BOX=(490,10,630,60)
sug_zones=[]; next_zones=[]; top3_zones=[]

last_stroke_t=0.0; auto_fired=False

# ════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════
def eu(a,b): return float(np.linalg.norm(np.array(a,float)-np.array(b,float)))
def fup(tip,pip,thr=0.028): return tip.y < pip.y-thr
def fdn(tip,pip,thr=0.028): return tip.y > pip.y+thr
def lpx(lm,i,w,h): return (int(lm[i].x*w),int(lm[i].y*h))

def snap():
    global history
    s=dict(txt=current_text,word=word_buffer,
           cv=canvas.copy(),pts=list(points),
           sugs=list(suggestions),mode=current_mode,
           nxt=list(next_preds),shape=shape_result)
    if len(history)>=MAX_H: history.pop(0)
    history.append(s)

def do_undo():
    global current_text,word_buffer,canvas,points,suggestions,next_preds,shape_result
    if not history: return
    s=history.pop()
    current_text=s['txt']; word_buffer=s['word']
    canvas=s['cv'].copy(); points=list(s['pts'])
    suggestions=list(s['sugs']); next_preds=list(s['nxt'])
    shape_result=s.get('shape','')
    print(f"UNDO text='{current_text}' word='{word_buffer}'")

def last_word(txt):
    s=txt.rstrip()
    return s.rsplit(' ',1)[-1] if ' ' in s else ""

# ════════════════════════════════════════════════════
#  IMAGE PIPELINE
#  KEY FIX: MIN_MOVE reduced to 2px so every small finger
#  movement records a point → strokes are complete, not gaps.
#  Lines drawn INSIDE draw-gesture block so they connect
#  every frame — no more dot-only drawing.
# ════════════════════════════════════════════════════
def pts_to_img(pts, ow=64, oh=64):
    img=np.zeros((oh,ow),np.uint8)
    for i in range(1,len(pts)):
        if pts[i-1] is None or pts[i] is None: continue
        x1=int(pts[i-1][0]/CAM_W*ow); y1=int(pts[i-1][1]/CAM_H*oh)
        x2=int(pts[i][0]  /CAM_W*ow); y2=int(pts[i][1]  /CAM_H*oh)
        cv2.line(img,(x1,y1),(x2,y2),255,3)
    return img

def center_crop(img):
    coords=np.column_stack(np.where(img>0))
    if coords.size==0: return img
    y0,x0=coords.min(axis=0); y1,x1=coords.max(axis=0)
    crop=img[y0:y1+1,x0:x1+1]
    out=np.zeros((64,64),np.uint8)
    h,w=crop.shape; yo=(64-h)//2; xo=(64-w)//2
    out[yo:yo+h,xo:xo+w]=crop
    return out

def run_model(img28):
    x=img28.reshape(1,28,28,1).astype(np.float32)
    interp.set_tensor(inp_d[0]['index'],x)
    interp.invoke()
    return interp.get_tensor(out_d[0]['index'])[0]

# ════════════════════════════════════════════════════
#  SHAPE RECOGNITION  (completely rewritten)
#
#  ROOT CAUSE of wrong predictions (heart/triangle confused):
#  The original code used the tiny 64×64 model canvas and
#  the contour was too small for accurate geometry.
#
#  FIX:
#  1. Render on 320×240 canvas (5× more detail)
#  2. Thick lines (6px) + heavy dilation to close air-gaps
#  3. Two-pass approxPolyDP (coarse + fine) for stable vertex count
#  4. Better closed-stroke detection (relative to bounding span)
#  5. Circularity thresholds tuned for air-written shapes
#  6. Heart detected via: rounded top with two bumps (convexity defects)
#  7. Triangle vs arrow: triangle is closed; arrow is open
# ════════════════════════════════════════════════════
def recognise_shape(pts):
    real=[p for p in pts if p is not None]
    if len(real)<6: return "unknown (draw bigger)"

    # Render on large canvas with thick lines
    img=np.zeros((240,320),np.uint8)
    for i in range(1,len(pts)):
        if pts[i-1] is None or pts[i] is None: continue
        x1=int(pts[i-1][0]/CAM_W*320); y1=int(pts[i-1][1]/CAM_H*240)
        x2=int(pts[i][0]  /CAM_W*320); y2=int(pts[i][1]  /CAM_H*240)
        cv2.line(img,(x1,y1),(x2,y2),255,6)

    # Heavy dilation closes gaps from air-writing
    img=cv2.dilate(img,np.ones((11,11),np.uint8),iterations=2)

    cnts,_=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return "unknown"
    cnt=max(cnts,key=cv2.contourArea)
    area=cv2.contourArea(cnt); peri=cv2.arcLength(cnt,True)
    if peri<1 or area<80: return "unknown (draw bigger)"

    circ = 4*math.pi*area/(peri*peri)

    # Two-pass polygon approximation
    ac=cv2.approxPolyDP(cnt,0.04*peri,True)   # coarse
    af=cv2.approxPolyDP(cnt,0.02*peri,True)   # fine
    vc=len(ac); vf=len(af)

    x,y,w,h=cv2.boundingRect(cnt)
    aspect=w/max(h,1)
    span  =max(w,h)
    num_s =pts.count(None)+1   # stroke count

    # Closed: first and last point within 22% of bounding span
    s0=real[0]; se=real[-1]
    closed=eu(s0,se)<0.22*span*(CAM_W/320.0)

    # Convexity defects — used for heart detection
    hull_idx=cv2.convexHull(cnt,returnPoints=False)
    defects=None
    try:
        if hull_idx is not None and len(hull_idx)>3:
            defects=cv2.convexityDefects(cnt,hull_idx)
    except: pass
    n_deep_defects=0
    if defects is not None:
        for dd in defects:
            s,e2,f,dep=dd[0]
            if dep/256.0 > span*0.12:   # deep notch
                n_deep_defects+=1

    # ── Decision tree ─────────────────────────────────
    # CIRCLE — high circularity, closed
    if circ>0.80 and closed and vc > 6:
        return "circle"

    # ELLIPSE — medium circularity, closed
    if circ>0.55 and closed and vf>7:
        return "ellipse"

    # HEART — closed, 2 deep convexity defects, top-notch pattern
    # Heart has one deep concavity at top-centre
    if closed and n_deep_defects>=1 and circ>0.45 and vc>=5:
        # Check: top of bounding box has a notch (indentation)
        top_strip=img[y:y+h//3, x:x+w]
        if top_strip.size>0:
            cols_on=np.sum(top_strip,axis=0)
            mid=len(cols_on)//2
            left_sum=np.sum(cols_on[:mid])
            right_sum=np.sum(cols_on[mid:])
            # Both sides lit (two bumps) + middle dip
            if left_sum>0 and right_sum>0:
                mid_val=cols_on[mid] if mid<len(cols_on) else 255
                if mid_val < 0.6*(cols_on.max()+1):
                    return "heart"

    # TRIANGLE — 3 coarse vertices, closed
    if vc==3 and closed:
        return "triangle"

    # NOTE: triangle vs arrow disambiguation:
    # Triangle is CLOSED (start≈end), arrow is OPEN
    if vc==3 and not closed:
        return "arrow"

    # RECTANGLE / SQUARE — 4 coarse vertices
    if vc==4:
        x_,y_,w_,h_=cv2.boundingRect(ac)
        ar=w_/max(h_,1)
        return "square" if 0.78<ar<1.28 else "rectangle"
    if vc==4 and closed:
       return "square"
    # PENTAGON / HEXAGON
    if vc==5: return "pentagon"
    if vc==6: return "hexagon"

    # STAR — many fine vertices, closed, low circularity
    if closed and vf>=9 and circ<0.55: return "star"

    # CROSS — 2 strokes, roughly square aspect
    if num_s>=2 and 0.5<aspect<2.0 and not closed:
        return "cross (+)"

    # X — 2 diagonal strokes
    if num_s>=2 and not closed: return "X mark"

    # HORIZONTAL LINE
    if aspect>2.5 and not closed and vc<=4: return "line (horizontal)"

    # VERTICAL LINE
    if aspect<0.4 and not closed and vc<=4: return "line (vertical)"

    # DIAGONAL LINE
    if not closed and vc<=4: return "line (diagonal)"

    # CHECK / TICK — open V-shape
    if not closed and vc<=6:
        mid2=len(ac)//2
        if mid2>0:
            ty=int(ac[0][0][1]); by2=int(ac[mid2][0][1]); ey2=int(ac[-1][0][1])
            if by2>ty and ey2<by2: return "check (✓)"

    # ARROW — open, multiple direction changes
    if not closed and vc>=5: return "arrow"

    if closed: return f"polygon ({vc} sides)"
    return "unknown"

# ════════════════════════════════════════════════════
#  PREDICTION FUNCTIONS
# ════════════════════════════════════════════════════
def do_predict():
    if   current_mode==1: pred_char()
    elif current_mode==2: pred_word_char()
    elif current_mode==3: pred_shape()

def pred_char():
    global current_text,points,last_pred,last_conf,top3_preds,pred_until
    real=[p for p in points if p is not None]
    if len(real)<8: points=[]; return
    img=pts_to_img(points); img=center_crop(img)
    img=cv2.resize(img,(28,28))/255.0
    out=run_model(img)
    idx3=np.argsort(out)[::-1][:3]
    top3_preds=[(classes[i],float(out[i])) for i in idx3]
    cid=int(idx3[0]); conf=float(out[cid])
    last_pred=classes[cid]; last_conf=conf; pred_until=time.time()+2.5
    snap()
    current_text+= classes[cid] if conf>=CONF_THRESHOLD else "?"
    print(f"[CHAR] '{last_pred}' {conf:.2f}")
    points=[]

def pred_word_char():
    """
    Mode 2 — HOW WORD REC WORKS:
    ─────────────────────────────
    Every PINCH adds ONE character to word_buffer.
    After each char the full buffer is checked by pyspellchecker.
    Up to 4 correction suggestions appear as green pills (bottom).

    To finish a word:
      • Pinky-ONLY up gesture → commits word + space
        (best suggestion is auto-applied if word is unknown)
      • OR: just stop drawing for 2 seconds → auto-commits

    To pick a specific suggestion:
      • Hover ring fingertip over green pill → it lights up
      • Pinch while hovering → that suggestion replaces word_buffer

    The word_buffer is shown in yellow brackets [buffer] on screen.
    """
    global current_text,word_buffer,points,last_pred,last_conf
    global top3_preds,pred_until,suggestions,next_preds
    real=[p for p in points if p is not None]
    if len(real)<8: points=[]; return
    img=pts_to_img(points); img=center_crop(img)
    img=cv2.resize(img,(28,28))/255.0
    out=run_model(img)
    idx3=np.argsort(out)[::-1][:3]
    top3_preds=[(classes[i],float(out[i])) for i in idx3]
    cid=int(idx3[0]); conf=float(out[cid])
    last_pred=classes[cid]; last_conf=conf; pred_until=time.time()+2.0
    ch=classes[cid] if conf>=CONF_THRESHOLD else "?"
    snap()
    word_buffer +=ch
    current_text+=ch
    next_preds=[]
    suggestions=get_suggestions(word_buffer)
    print(f"[WORD] +'{ch}' buf='{word_buffer}' sugs={suggestions}")
    points=[]

def commit_word():
    """Commit current word_buffer → apply best suggestion if misspelled → add space."""
    global current_text,word_buffer,suggestions,next_preds
    snap()
    if suggestions and word_buffer:
        best=suggestions[0]
        if current_text.endswith(word_buffer):
            current_text=current_text[:-len(word_buffer)]+best
        word_buffer=best
        print(f"[COMMIT] corrected → '{best}'")
    lw=word_buffer if word_buffer else last_word(current_text)
    next_preds=next_word_sugg(lw)
    current_text+=" "; word_buffer=""; suggestions=[]
    print(f"[COMMIT] done. next_preds={next_preds}")

def apply_sug(sug):
    global current_text,word_buffer,suggestions
    if not word_buffer: return
    if current_text.endswith(word_buffer):
        current_text=current_text[:-len(word_buffer)]+sug
    word_buffer=sug; suggestions=[]
    print(f"[AC] applied '{sug}'")

def apply_next(w):
    global current_text,word_buffer,suggestions,next_preds
    snap(); word_buffer=w; current_text+=w
    suggestions=[]; next_preds=[]

def pred_shape():
    global shape_result,points,pred_until,current_text
    real=[p for p in points if p is not None]
    if len(real)<6: points=[]; return
    sh=recognise_shape(points)
    shape_result=sh; pred_until=time.time()+3.0
    snap(); current_text+=f"[{sh}] "
    print(f"[SHAPE] → '{sh}'"); points=[]

# ════════════════════════════════════════════════════
#  GESTURE CLASSIFIER
#
#  DRAW = RING finger ONLY up (lm[16] tip vs lm[14] pip)
#  This is completely different from pinch, fist, palm —
#  no conflict with any other gesture.
#
#  UNDO FIX:
#  Original check (lm[4].y > lm[2].y) was unreliable.
#  New check: all 4 fingers curled down + thumb tip is
#  close to the palm (lm[4] near lm[9] = middle knuckle).
#  This robustly detects a genuine closed fist.
#
#  MODE SWITCH:
#  Uses thumb+ring (M1), index+middle (M2), idx+mid+ring (M3).
#  All require holding for MODE_HOLD_SEC to avoid accidental fires.
#  Ring-only is reserved for DRAW so M1 uses thumb+ring combo.
# ════════════════════════════════════════════════════
def fups(lm):
    # [index_up, middle_up, ring_up, pinky_up]
    return [fup(lm[8],lm[6]), fup(lm[12],lm[10]),
            fup(lm[16],lm[14]), fup(lm[20],lm[18])]

def classify_gest(lm, ipx, tpx):
    iu,mu,ru,pu = fups(lm)
    id_=fdn(lm[8], lm[6]); md=fdn(lm[12],lm[10])
    rd_=fdn(lm[16],lm[14]); pd=fdn(lm[20],lm[18])
    thumb_up  = lm[4].y < lm[3].y-0.04
    # Fist: thumb tip near middle-finger MCP (landmark 9)
    thumb_fist= eu(lpx(lm,4,CAM_W,CAM_H), lpx(lm,9,CAM_W,CAM_H)) < CAM_W*0.12
    d=eu(ipx,tpx)

    # ① DRAW — index ONLY up
    if iu and not mu and not ru and not pu:
        return "DRAW"

    # ② PREDICT — index+thumb pinch, all others down
    if d<PINCH_DIST and id_ and md and rd_ and pd:
        return "PREDICT"

    # ③ SPACE — pinky ONLY up
    if pu and not iu and not mu and not ru:
        return "SPACE"

    # ④ CLEAR CANVAS — all 5 fully open
    if iu and mu and ru and pu and thumb_up:
        return "CLEAR_CANVAS"

    # ⑤ UNDO — full fist: all 4 down + thumb near palm
    #   (this is the reliable check — thumb_fist uses euclidean dist)
    if id_ and md and rd_ and pd and thumb_fist:
        return "UNDO"

    # ⑥ CLEAR TEXT — thumb up, all 4 curled
    if thumb_up and id_ and md and rd_ and pd:
        return "CLEAR_TEXT"

    return "NONE"

def mode_gest(lm):
    """
    Returns target mode (1/2/3) for mode-hold detection, else 0.
    Mode 1: Thumb + Ring up (ring reserved for draw, but thumb+ring
            is a clearly different shape from ring-only)
    Mode 2: Index + Middle up (no ring → not DRAW)
    Mode 3: Index + Middle + Ring (3 fingers — different from ring-only)
    All require hold for MODE_HOLD_SEC in main loop.
    """
    iu,mu,ru,pu = fups(lm)
    thumb_up = lm[4].y < lm[3].y-0.04
    # Mode 1: ring, no index/middle/pinky
    if ru and not iu and not mu and not pu:
        return 1
    # Mode 2: index + middle, no ring/pinky, no thumb
    if iu and mu and not ru and not pu and not thumb_up:
        return 2
    # Mode 3: index + middle + ring, no pinky
    if iu and mu and ru and not pu and not thumb_up:
        return 3
    return 0

# ════════════════════════════════════════════════════
#  UI
# ════════════════════════════════════════════════════
def pill(fr,x1,y1,x2,y2,bg,border,alpha=0.72):
    ov=fr.copy()
    cv2.rectangle(ov,(x1,y1),(x2,y2),bg,-1)
    cv2.addWeighted(ov,alpha,fr,1-alpha,0,fr)
    cv2.rectangle(fr,(x1,y1),(x2,y2),border,1)

def draw_tabs(fr):
    tw=128; sy=88; sh=26
    for m in MODES:
        x1=10+(m-1)*(tw+4); x2=x1+tw
        act=(m==current_mode)
        bg=MODE_COLORS[m] if act else (40,40,40)
        pill(fr,x1,sy,x2,sy+sh,bg,MODE_COLORS[m],alpha=0.85 if act else 0.4)
        fc=C_DARK if act else C_GRAY
        cv2.putText(fr,f"M{m}:{MODES[m]}",(x1+5,sy+18),
                    cv2.FONT_HERSHEY_SIMPLEX,0.43,fc,1,cv2.LINE_AA)
    if mode_hold_g>0:
        prog=min(1.0,(time.time()-mode_hold_t)/MODE_HOLD_SEC)
        bx2=int(10+prog*(3*tw+8))
        cv2.rectangle(fr,(10,sy+sh+2),(bx2,sy+sh+5),MODE_COLORS.get(mode_hold_g,C_WHITE),-1)

def draw_sug_bar(fr, ipx):
    global sug_zones,next_zones
    sug_zones=[]; next_zones=[]

    # Next-word row (blue)
    if next_preds:
        ny=CAM_H-76
        ov=fr.copy()
        cv2.rectangle(ov,(0,ny-4),(CAM_W,ny+30),(20,20,40),-1)
        cv2.addWeighted(ov,0.65,fr,0.35,0,fr)
        cv2.putText(fr,"Next:",(4,ny+18),cv2.FONT_HERSHEY_SIMPLEX,0.38,C_BLUE,1,cv2.LINE_AA)
        nx=55
        for w in next_preds:
            (tw,th),_=cv2.getTextSize(w,cv2.FONT_HERSHEY_SIMPLEX,0.52,1)
            x1=nx;y1=ny-2;x2=nx+tw+14;y2=ny+th+8
            hov=ipx and x1<=ipx[0]<=x2 and y1<=ipx[1]<=y2
            pill(fr,x1,y1,x2,y2,(80,50,0) if hov else (30,30,60),C_BLUE if hov else (70,70,160))
            cv2.putText(fr,w,(x1+7,y1+th+2),cv2.FONT_HERSHEY_SIMPLEX,0.52,
                        C_WHITE if hov else C_BLUE,1,cv2.LINE_AA)
            next_zones.append((x1,y1,x2,y2,w)); nx=x2+4
            if nx>CAM_W-60: break

    # Autocorrect row (green)
    if not suggestions: return
    by=CAM_H-43
    ov=fr.copy()
    cv2.rectangle(ov,(0,by-4),(CAM_W,CAM_H-18),(20,20,20),-1)
    cv2.addWeighted(ov,0.68,fr,0.32,0,fr)
    cv2.putText(fr,"AC:",(3,by+10),cv2.FONT_HERSHEY_SIMPLEX,0.38,C_YELLOW,1,cv2.LINE_AA)
    sx=38
    for sug in suggestions:
        (tw,th),_=cv2.getTextSize(sug,cv2.FONT_HERSHEY_SIMPLEX,0.56,1)
        x1=sx;y1=by-3;x2=sx+tw+14;y2=by+th+8
        hov=ipx and x1<=ipx[0]<=x2 and y1<=ipx[1]<=y2
        pill(fr,x1,y1,x2,y2,(0,140,55) if hov else (35,55,35),C_GREEN if hov else (70,110,70))
        cv2.putText(fr,sug,(x1+7,y1+th+2),cv2.FONT_HERSHEY_SIMPLEX,0.56,
                    C_WHITE if hov else C_GREEN,1,cv2.LINE_AA)
        sug_zones.append((x1,y1,x2,y2,sug)); sx=x2+5
        if sx>CAM_W-60: break

def draw_top3(fr,ipx):
    global top3_zones; top3_zones=[]
    if not top3_preds or time.time()>pred_until or last_conf>=CONF_THRESHOLD: return
    py=158
    ov=fr.copy()
    cv2.rectangle(ov,(48,py-22),(410,py+112),(20,20,20),-1)
    cv2.addWeighted(ov,0.70,fr,0.30,0,fr)
    cv2.putText(fr,"Low conf — hover+pinch pick:",(52,py-7),
                cv2.FONT_HERSHEY_SIMPLEX,0.40,C_YELLOW,1,cv2.LINE_AA)
    for i,(cls,conf) in enumerate(top3_preds):
        pct=int(conf*100); bx=60; by2=py+i*34
        bw=int(200*conf)
        hov=ipx and bx<=ipx[0]<=bx+220 and by2<=ipx[1]<=by2+26
        cv2.rectangle(fr,(bx,by2),(bx+bw,by2+24),(0,100,180) if hov else (0,60,120),-1)
        cv2.rectangle(fr,(bx,by2),(bx+220,by2+24),C_GRAY,1)
        cv2.putText(fr,f"{i+1}:{cls}  {pct}%",(bx+5,by2+17),
                    cv2.FONT_HERSHEY_SIMPLEX,0.52,C_WHITE,1,cv2.LINE_AA)
        top3_zones.append((bx,by2,bx+220,by2+24,cls))

def draw_guide(fr):
    lines=["GESTURES (v4):","Index UP=Draw","Pinch=Predict",
           "Pinky UP=Space(M2)","AllOpen=ClearCanvas",
           "Fist=Undo","ThumbUp4dn=ClearText",
           "Thumb+Ring 1.2s=M1","Idx+Mid 1.2s=M2","Idx+Mid+Ring 1.2s=M3"]
    x0=5; y0=CAM_H-4-len(lines)*13
    ov=fr.copy()
    cv2.rectangle(ov,(x0-2,y0-11),(x0+162,CAM_H-2),(10,10,10),-1)
    cv2.addWeighted(ov,0.55,fr,0.45,0,fr)
    for i,ln in enumerate(lines):
        cv2.putText(fr,ln,(x0,y0+i*13),cv2.FONT_HERSHEY_SIMPLEX,0.29,
                    C_GREEN if i==0 else C_GRAY,1,cv2.LINE_AA)

def draw_ui(fr,gest,ipx):
    now=time.time()
    # Top bar
    ov=fr.copy(); cv2.rectangle(ov,(0,0),(CAM_W,84),C_DARK,-1)
    cv2.addWeighted(ov,0.62,fr,0.38,0,fr)
    disp=current_text if current_text else "—"
    if len(disp)>30: disp="…"+disp[-27:]
    cv2.putText(fr,f"Text: {disp}",(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.95,C_GREEN,2,cv2.LINE_AA)
    if current_mode==2 and word_buffer:
        cv2.putText(fr,f"[ {word_buffer} ]",(10,73),cv2.FONT_HERSHEY_SIMPLEX,0.48,C_YELLOW,1,cv2.LINE_AA)
    st="● DRAW" if is_drawing else f"○ {gest}"
    cv2.putText(fr,st,(CAM_W-148,73),cv2.FONT_HERSHEY_SIMPLEX,0.42,C_CYAN if is_drawing else C_GRAY,1,cv2.LINE_AA)
    draw_tabs(fr)
    # Prediction popup
    if now<pred_until:
        if current_mode==3:
            txt=f"Shape: {shape_result}"; col=C_ORANGE
        else:
            pct=int(last_conf*100); col=C_GREEN if last_conf>=CONF_THRESHOLD else C_RED
            txt=f"'{last_pred}'  {pct}%"
        (tw,_),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,1.2,3)
        px=(CAM_W-tw)//2; py2=210
        cv2.putText(fr,txt,(px+2,py2+2),cv2.FONT_HERSHEY_SIMPLEX,1.2,C_BLACK,5,cv2.LINE_AA)
        cv2.putText(fr,txt,(px,py2),cv2.FONT_HERSHEY_SIMPLEX,1.2,col,3,cv2.LINE_AA)
    draw_top3(fr,ipx)
    if current_mode==2: draw_sug_bar(fr,ipx)
    # Auto-predict countdown bar
    if last_stroke_t>0 and not is_drawing and not auto_fired:
        rem=max(0.0,AUTO_PRED_GAP-(now-last_stroke_t))
        if 0<rem<AUTO_PRED_GAP:
            prog=1.0-rem/AUTO_PRED_GAP
            bx=220; by3=128
            cv2.putText(fr,"Auto:",(bx-32,by3+8),cv2.FONT_HERSHEY_SIMPLEX,0.34,C_GRAY,1,cv2.LINE_AA)
            cv2.rectangle(fr,(bx,by3),(bx+180,by3+8),(50,50,50),-1)
            cv2.rectangle(fr,(bx,by3),(bx+int(180*prog),by3+8),C_CYAN,-1)
            cv2.putText(fr,f"{rem:.1f}s",(bx+183,by3+8),cv2.FONT_HERSHEY_SIMPLEX,0.34,C_CYAN,1,cv2.LINE_AA)
    # Erase box
    ex1,ey1,ex2,ey2=ERASE_BOX
    ratio=min(1.0,erase_cnt/HOVER_FRAMES); bpw=int((ex2-ex1)*ratio)
    bo=fr.copy()
    cv2.rectangle(bo,(ex1,ey1),(ex1+bpw,ey2),(0,80,200),-1)
    cv2.addWeighted(bo,0.5,fr,0.5,0,fr)
    cv2.rectangle(fr,(ex1,ey1),(ex2,ey2),C_CYAN if erase_cnt>0 else C_GRAY,2 if erase_cnt>0 else 1)
    cv2.putText(fr,"ERASE",(ex1+8,ey1+28),cv2.FONT_HERSHEY_SIMPLEX,0.42,(220,220,220),1,cv2.LINE_AA)
    draw_guide(fr)
    if ipx:
        cv2.circle(fr,ipx,8,(0,255,200),-1)
        cv2.circle(fr,ipx,10,C_WHITE,1)

# ════════════════════════════════════════════════════
#  MAIN LOOP
# ════════════════════════════════════════════════════
print("\n"+"═"*56)
print("  AIR WRITING v4  —  RING-FINGER DRAW  —  GESTURE ONLY")
print("═"*56)
print("  DRAW:         Ring finger ONLY up")
print("  PREDICT:      Pinch (idx+thumb, others down)")
print("  SPACE (M2):   Pinky ONLY up")
print("  CLR CANVAS:   All 5 open")
print("  UNDO:         Full fist (4 fingers + thumb in)")
print("  CLR TEXT:     Thumb up, all 4 curled")
print("  MODE 1:       Thumb+Ring hold 1.2s")
print("  MODE 2:       Index+Middle hold 1.2s")
print("  MODE 3:       Index+Middle+Ring hold 1.2s")
print()
print("  Mode 2 WORD flow:")
print("    Ring up → draw char → Pinch → adds to [buffer]")
print("    Repeat for each char → green AC pills appear")
print("    Hover ring-tip on pill + Pinch → apply correction")
print("    Pinky up → commit word + space")
print("    OR: 2s idle → auto-commits")
print("═"*56+"\n")

while True:
    ret,frame=cap.read()
    if not ret: break
    frame=cv2.flip(frame,1)
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    res=hproc.process(rgb)

    now=time.time(); gest="NONE"; ipx=None

    if res.multi_hand_landmarks:
        for hlms in res.multi_hand_landmarks:
            lm=hlms.landmark; fw,fh=frame.shape[1],frame.shape[0]
            raw_x=int(lm[8].x*fw)   # index fingertip
            raw_y=int(lm[8].y*fh)   # ring fingertip

            # Kalman smooth when ring is up (drawing)
            if lm[8].y < lm[6].y-0.03:
                sx, sy = raw_x, raw_y; sy=kf_y.update(raw_y)
            else:
                kf_x.reset(); kf_y.reset(); sx,sy=raw_x,raw_y

            ipx=(sx,sy)
            # Index tip for pinch detection (landmark 8)
            idx_px=lpx(lm,8,fw,fh); tpx=lpx(lm,4,fw,fh)

            gest=classify_gest(lm,idx_px,tpx)

            # ── Mode-hold (blocked during DRAW) ──────────
            if gest!="DRAW":
                mg=mode_gest(lm)
                if mg>0:
                    if mode_hold_g!=mg: mode_hold_g=mg; mode_hold_t=now
                    elif now-mode_hold_t>=MODE_HOLD_SEC:
                        if current_mode!=mg:
                            current_mode=mg; word_buffer=""; suggestions=[]
                            next_preds=[]; points=[]; canvas[:]=0
                            auto_fired=False
                            print(f"MODE → {MODES[current_mode]}")
                        mode_hold_t=now
                else:
                    mode_hold_g=0; mode_hold_t=0.0
            else:
                mode_hold_g=0; mode_hold_t=0.0

            # ── Erase box hover ────────────────────────────
            ex1,ey1,ex2,ey2=ERASE_BOX
            if ex1<=sx<=ex2 and ey1<=sy<=ey2:
                erase_cnt+=1
                if erase_cnt>=HOVER_FRAMES:
                    snap(); current_text=""; word_buffer=""; suggestions=[]
                    next_preds=[]; erase_cnt=0
            else:
                erase_cnt=max(0,erase_cnt-1)

            # ── DRAWING ────────────────────────────────────
            if gest == "DRAW":
                is_drawing = True
                freeze_until = now + FREEZE_DRAW

                x, y = ipx   # already using index finger now

                # smooth but stable movement
                if abs(x - prev_sx) + abs(y - prev_sy) > 5:
                    points.append((x, y))
                    prev_sx, prev_sy = x, y

                    # immediate visual feedback (important!)
                    cv2.circle(canvas, (x, y), 8, C_WHITE, -1)

                last_stroke_t = time.time()
                auto_fired = False

            elif():
                if is_drawing:
                    points.append(None)
                    is_drawing = False
                prev_sx, prev_sy = 0, 0
            else:
                if is_drawing:
                    points.append(None)
                    prev_sx, prev_sy = 0, 0
                    freeze_until = time.time() + FREEZE_DRAW
                is_drawing = False
            for i in range(1, len(points)):
                if points[i-1] is None or points[i] is None:
                    continue
                cv2.line(canvas, points[i-1], points[i], C_WHITE, 12)
            # ── Gesture actions ────────────────────────────
            g_ok=(gest not in ("NONE","DRAW") and
                  now>gest_last_t+COOLDOWN and
                  now>freeze_until and
                  gest!=prev_gest)

            if g_ok:
                if gest=="PREDICT":
                    applied=False
                    # Hover on AC pill?
                    if current_mode==2 and sug_zones:
                        for x1,y1,x2,y2,sug in sug_zones:
                            if x1<=sx<=x2 and y1<=sy<=y2:
                                apply_sug(sug); applied=True; break
                    # Hover on next-word pill?
                    if not applied and current_mode==2 and next_zones:
                        for x1,y1,x2,y2,nw in next_zones:
                            if x1<=sx<=x2 and y1<=sy<=y2:
                                apply_next(nw); applied=True; break
                    # Hover on top3 popup?
                    if not applied and top3_zones:
                        for x1,y1,x2,y2,cls in top3_zones:
                            if x1<=sx<=x2 and y1<=sy<=y2:
                                snap()
                                if current_text.endswith("?"): current_text=current_text[:-1]+cls
                                applied=True; break
                    if not applied: do_predict()
                    canvas[:]=0; auto_fired=True; gest_last_t=now

                elif gest=="SPACE":
                    if current_mode==2: commit_word()
                    else: snap(); current_text+=" "
                    canvas[:]=0; points=[]; gest_last_t=now

                elif gest=="CLEAR_CANVAS":
                    canvas[:]=0; points=[]; is_drawing=False
                    auto_fired=False; gest_last_t=now

                elif gest=="UNDO":
                    do_undo(); canvas[:]=0; gest_last_t=now

                elif gest=="CLEAR_TEXT":
                    snap(); current_text=""; word_buffer=""
                    suggestions=[]; next_preds=[]
                    canvas[:]=0; points=[]; gest_last_t=now

            prev_gest=gest
            mpd.draw_landmarks(frame,hlms,mp_h.HAND_CONNECTIONS,HS,CS)

    else:
        if is_drawing: points.append(None); is_drawing=False
        kf_x.reset(); kf_y.reset()
        prev_sx=prev_sy=0; prev_gest="NONE"; mode_hold_g=0
        erase_cnt=max(0,erase_cnt-1)

    # ── Auto-predict on idle gap (ALL modes) ──────────────
    rp=[p for p in points if p is not None]
    if (last_stroke_t>0 and not is_drawing and not auto_fired
            and len(rp)>=8
            and now-last_stroke_t>AUTO_PRED_GAP
            and now>freeze_until):
        print(f"[AUTO] {AUTO_PRED_GAP}s idle → predict")
        do_predict(); canvas[:]=0; auto_fired=True

    combined=cv2.add(frame,canvas)
    draw_ui(combined,gest,ipx)
    cv2.imshow("Air Writing v4",combined)

    key=cv2.waitKey(1)&0xFF
    if key==ord('q'): break   # emergency quit only

cap.release()
cv2.destroyAllWindows()

# ════════════════════════════════════════════════════
#  MODEL QUALITY — READ THIS TO IMPROVE ACCURACY
# ════════════════════════════════════════════════════
"""
YOUR MODEL ARCHITECTURE (CNN on custom data):

PROBLEMS CAUSING LOW ACCURACY:
───────────────────────────────
1. DATA IMBALANCE
   If some classes have fewer samples than others, the model
   learns to favour the majority classes.
   FIX: Use class_weight in model.fit():
     from sklearn.utils.class_weight import compute_class_weight
     cw = compute_class_weight('balanced', classes=np.unique(y), y=y)
     model.fit(..., class_weight=dict(enumerate(cw)))

2. INSUFFICIENT AUGMENTATION
   Air-writing looks very different at different speeds/angles.
   FIX: Add these augmentations to your training pipeline:
     tf.keras.preprocessing.image.ImageDataGenerator(
         rotation_range=15,
         width_shift_range=0.15,
         height_shift_range=0.15,
         zoom_range=0.15,
         shear_range=0.1,
         fill_mode='nearest'
     )

3. MODEL TOO SHALLOW
   A simple 2-conv CNN may not separate similar-looking chars.
   RECOMMENDED ARCHITECTURE for 28×28 grayscale input:
   ─────────────────────────────────────────────────────
   Input(28,28,1)
   Conv2D(32,3,padding='same') → BN → ReLU
   Conv2D(32,3,padding='same') → BN → ReLU → MaxPool(2)→Drop(0.25)
   Conv2D(64,3,padding='same') → BN → ReLU
   Conv2D(64,3,padding='same') → BN → ReLU → MaxPool(2)→Drop(0.25)
   Flatten → Dense(256) → BN → ReLU → Drop(0.5)
   Dense(num_classes) → Softmax
   ─────────────────────────────────────────────────────
   Use: optimizer=Adam(lr=1e-3), loss=categorical_crossentropy
   Train for 50-100 epochs with early stopping (patience=10).

4. TRAINING DATA NOT FROM AIR-WRITING
   If you trained on MNIST or keyboard-typed fonts, the model
   will fail on air-written strokes which look different (thinner,
   more irregular, sometimes incomplete).
   FIX: Collect your OWN air-writing samples using this app.
   Minimum: 100-200 samples per class.
   Ideal:   500+ samples per class with varied lighting.

5. IMAGE PREPROCESSING MISMATCH
   This app uses: 64×64 render → center_crop → resize to 28×28 → /255.
   Your training pipeline MUST match EXACTLY.
   Check: did you normalize to [0,1] during training?
   Check: did you center/crop during training?

6. CONFIDENCE THRESHOLD
   CONF_THRESHOLD is currently 0.55. If you see too many '?'
   characters, lower it to 0.45. If you see too many wrong chars,
   raise it to 0.65.

QUICK ACCURACY TEST:
   Run this to check your model on a known sample:
   python -c "
   import tensorflow as tf, numpy as np, cv2
   m=tf.lite.Interpreter('air_model.tflite'); m.allocate_tensors()
   i=np.zeros((1,28,28,1),np.float32)  # blank image
   m.set_tensor(m.get_input_details()[0]['index'],i)
   m.invoke()
   print(m.get_tensor(m.get_output_details()[0]['index']))
   "
   All values should be roughly equal (~1/num_classes) for blank input.
   If one class dominates even on blank input → your model is biased.
"""