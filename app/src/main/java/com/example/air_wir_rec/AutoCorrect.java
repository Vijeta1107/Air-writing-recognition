package com.example.air_wir_rec;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * AutoCorrect — Fixed version.
 *
 * KEY FIXES:
 * 1. getSuggestions() always lowercases input before matching.
 *    This is critical: the user writes "T" (uppercase) but the dictionary
 *    has "the", "to", "them". Without lowercase conversion, no match found.
 *
 * 2. bestSuggestion() only triggers for words >= 3 chars to avoid
 *    aggressively replacing single or double letter inputs.
 *
 * 3. Prefix matching is Phase A (shown first), fuzzy is Phase B (fallback).
 *    This ensures "t" shows "the, to, them, thank, time" etc. immediately.
 *
 * 4. Extended bigram table for more common word pairs.
 *
 * 5. Extended dictionary with more common words.
 */
public class AutoCorrect {

    // ── Bigrams (next-word predictions) ──────────────────────────────
    private static final Map<String, String[]> BIGRAMS = new HashMap<>();
    static {
        BIGRAMS.put("i",      new String[]{"am","will","have","can","need","want","think","know"});
        BIGRAMS.put("you",    new String[]{"are","can","will","have","need","want","should"});
        BIGRAMS.put("he",     new String[]{"is","was","will","can","has","said","went"});
        BIGRAMS.put("she",    new String[]{"is","was","will","can","has","said","went"});
        BIGRAMS.put("we",     new String[]{"are","will","have","can","should","need"});
        BIGRAMS.put("they",   new String[]{"are","will","have","can","were","said"});
        BIGRAMS.put("the",    new String[]{"best","most","same","first","last","new","only"});
        BIGRAMS.put("a",      new String[]{"great","good","new","big","small","lot","few"});
        BIGRAMS.put("this",   new String[]{"is","was","will","can","has","means","looks"});
        BIGRAMS.put("my",     new String[]{"name","phone","home","friend","work","family","life"});
        BIGRAMS.put("please", new String[]{"help","send","call","come","wait","let","do"});
        BIGRAMS.put("can",    new String[]{"you","we","help","go","come","see","do"});
        BIGRAMS.put("how",    new String[]{"are","do","can","is","was","much","many"});
        BIGRAMS.put("what",   new String[]{"is","are","do","can","was","time","about"});
        BIGRAMS.put("good",   new String[]{"morning","night","work","job","day","luck","time"});
        BIGRAMS.put("hello",  new String[]{"there","world","friend","how","everyone"});
        BIGRAMS.put("thank",  new String[]{"you","god","him","her","them","everyone"});
        BIGRAMS.put("need",   new String[]{"help","more","to","your","some","a","the"});
        BIGRAMS.put("go",     new String[]{"to","ahead","now","there","back","home"});
        BIGRAMS.put("call",   new String[]{"me","you","him","her","back","later","now"});
        BIGRAMS.put("help",   new String[]{"me","you","please","him","her","us"});
        BIGRAMS.put("ok",     new String[]{"great","good","sure","fine","thanks","got"});
        BIGRAMS.put("hi",     new String[]{"there","how","friend","all","everyone"});
        BIGRAMS.put("see",    new String[]{"you","the","me","later","now","what","that"});
        BIGRAMS.put("do",     new String[]{"it","you","we","that","this","not","want"});
        BIGRAMS.put("is",     new String[]{"this","that","it","there","he","she","not"});
        BIGRAMS.put("have",   new String[]{"you","we","they","a","to","got","been"});
        BIGRAMS.put("tell",   new String[]{"me","you","him","her","them","us","that"});
        BIGRAMS.put("show",   new String[]{"me","it","you","them","us","how","where"});
        BIGRAMS.put("let",    new String[]{"me","it","go","him","us","know","see"});
        BIGRAMS.put("not",    new String[]{"sure","now","yet","here","good","really","yet"});
        BIGRAMS.put("come",   new String[]{"here","back","on","with","to","over"});
        BIGRAMS.put("take",   new String[]{"care","it","me","that","your","time"});
        BIGRAMS.put("make",   new String[]{"it","sure","me","that","a","the"});
        BIGRAMS.put("get",    new String[]{"me","it","back","up","out","ready","going"});
        BIGRAMS.put("want",   new String[]{"to","you","me","it","more","some"});
        BIGRAMS.put("know",   new String[]{"that","what","how","you","it","where","why"});
        BIGRAMS.put("think",  new String[]{"so","about","that","you","it","we","i"});
        BIGRAMS.put("time",   new String[]{"to","for","is","now","out","flies","zones"});
        BIGRAMS.put("back",   new String[]{"to","up","in","at","soon","home","later"});
        BIGRAMS.put("no",     new String[]{"more","one","way","time","thanks","problem"});
        BIGRAMS.put("all",    new String[]{"right","good","done","set","clear","the","in"});
        BIGRAMS.put("hey",    new String[]{"there","you","what","how","man","friend"});
        BIGRAMS.put("that",   new String[]{"is","was","will","can","has","would","should"});
        BIGRAMS.put("with",   new String[]{"you","me","him","her","them","us","the"});
        BIGRAMS.put("for",    new String[]{"you","me","the","a","that","this","now"});
    }

    // ── Dictionary (ranked by frequency, index 0 = most frequent) ───
    private static final String[] DICTIONARY = {
            // Ultra-common words
            "the","be","to","of","and","a","in","that","have","it",
            "for","not","on","with","he","as","you","do","at","this",
            "but","his","by","from","they","we","say","her","she","or",
            "an","will","my","one","all","would","there","their","what",
            "so","up","out","if","about","who","get","which","go","me",
            "when","make","can","like","time","no","just","him","know",
            "take","people","into","year","your","good","some","could",
            "them","see","other","than","then","now","look","only","come",
            "its","over","think","also","back","after","use","two","how",
            "our","work","first","well","way","even","new","want","because",
            "any","these","give","day","most","us","am","is","was","are",
            "were","has","had","did","does","done","been","being","having",
            // Common conversation words
            "great","between","need","large","often","hand","high","place",
            "hold","turn","help","start","hello","hi","hey","bye","yes",
            "ok","okay","sure","please","thank","thanks","sorry","excuse",
            "pardon","wait","stop","here","where","why","name","phone",
            "home","school","food","water","today","tomorrow","yesterday",
            "morning","afternoon","evening","night","right","left","next",
            "last","more","less","much","many","few","every","never",
            "always","sometimes","maybe","perhaps","really","very","quite",
            "too","just","again","still","already","soon","later","send",
            "call","text","email","message","write","read","draw","show",
            "tell","open","close","save","delete","create","find","search",
            "play","pause","going","coming","saying","seeing","away","down",
            // Words that are commonly confused / misrecognized
            "hello","help","here","have","has","had","him","his","her",
            "that","then","than","them","they","this","the","there","these",
            "what","when","where","which","while","who","why","with",
            "want","was","were","well","will","would",
            // Days
            "monday","tuesday","wednesday","thursday","friday","saturday","sunday",
            // Months
            "january","february","march","april","may","june","july",
            "august","september","october","november","december",
            // Numbers as words
            "zero","one","two","three","four","five","six","seven","eight","nine","ten",
            // Tech
            "android","google","computer","internet","app","phone","screen",
    };

    private static final Map<String, Integer> WORD_FREQ = new HashMap<>();
    static {
        for (int i = 0; i < DICTIONARY.length; i++) {
            // Don't overwrite — keep first (higher frequency) occurrence
            if (!WORD_FREQ.containsKey(DICTIONARY[i])) {
                WORD_FREQ.put(DICTIONARY[i], DICTIONARY.length - i);
            }
        }
    }

    // ── PUBLIC API ────────────────────────────────────────────────────

    /**
     * getSuggestions() — returns up to n suggestions for the partial word.
     *
     * FIX: Input is lowercased before lookup. Phase A = prefix matches
     * (shown first, sorted by frequency). Phase B = fuzzy fallback
     * (only when input >= 3 chars).
     *
     * Example: input "T" or "t" → lowercased to "t" → prefix matches:
     *   "the", "to", "that", "them", "then", "than", "they", "time", "thank", "think"...
     */
    public static List<String> getSuggestions(String word, int n) {
        if (word == null || word.isEmpty()) return new ArrayList<>();

        // FIX: Always lowercase before lookup
        String w = word.toLowerCase().trim();
        if (w.isEmpty()) return new ArrayList<>();

        List<String> prefixMatches = new ArrayList<>();
        List<String[]> fuzzyMatches = new ArrayList<>();

        for (String dict : DICTIONARY) {
            if (dict.startsWith(w)) {
                // Phase A: prefix match
                if (!prefixMatches.contains(dict)) {
                    prefixMatches.add(dict);
                }
            } else if (w.length() >= 3) {
                // Phase B: fuzzy only for 3+ char inputs
                if (Math.abs(dict.length() - w.length()) <= 2) {
                    int ed = editDistance(w, dict);
                    if (ed <= 2) {
                        int freq  = WORD_FREQ.getOrDefault(dict, 0);
                        int score = (2 - ed) * 1000 + freq;
                        fuzzyMatches.add(new String[]{dict, String.valueOf(score)});
                    }
                }
            }
        }

        // Sort prefix matches by frequency (higher = better)
        prefixMatches.sort((a, b) ->
                Integer.compare(
                        WORD_FREQ.getOrDefault(b, 0),
                        WORD_FREQ.getOrDefault(a, 0)));

        // Sort fuzzy matches by score
        fuzzyMatches.sort((a, b) ->
                Integer.compare(Integer.parseInt(b[1]), Integer.parseInt(a[1])));

        // Combine: prefix first, then fuzzy (no duplicates)
        List<String> result = new ArrayList<>(prefixMatches);
        for (String[] fm : fuzzyMatches) {
            if (!result.contains(fm[0])) result.add(fm[0]);
            if (result.size() >= n * 2) break;
        }

        return result.subList(0, Math.min(n, result.size()));
    }

    /**
     * bestSuggestion() — returns the best auto-correction for a word,
     * or null if the word is already correct or no good match found.
     *
     * Only triggers for words >= 3 chars to avoid replacing "A" or "to" etc.
     */
    public static String bestSuggestion(String word) {
        if (word == null || word.length() < 3) return null;
        String w = word.toLowerCase().trim();
        // Exact match in dictionary = no correction needed
        if (WORD_FREQ.containsKey(w)) return null;
        List<String> sugs = getSuggestions(w, 1);
        if (sugs.isEmpty()) return null;
        String best = sugs.get(0);
        int dist = editDistance(w, best);
        // Allow edit distance 1 for short words (3-4 chars), distance 2 for longer
        int maxAllowed = (w.length() >= 5) ? 2 : 1;
        return (dist <= maxAllowed) ? best : null;
    }

    /**
     * nextWordSuggestions() — bigram-based next word predictions.
     * Call this after the user commits a word (SPACE gesture).
     */
    public static List<String> nextWordSuggestions(String prevWord, int n) {
        if (prevWord == null) return new ArrayList<>();
        String[] preds = BIGRAMS.get(prevWord.toLowerCase().trim());
        if (preds == null) return new ArrayList<>();
        List<String> result = new ArrayList<>();
        for (int i = 0; i < Math.min(n, preds.length); i++) result.add(preds[i]);
        return result;
    }

    // ── Levenshtein edit distance ─────────────────────────────────────
    private static int editDistance(String a, String b) {
        int la = a.length(), lb = b.length();
        int[][] dp = new int[la + 1][lb + 1];
        for (int i = 0; i <= la; i++) dp[i][0] = i;
        for (int j = 0; j <= lb; j++) dp[0][j] = j;
        for (int i = 1; i <= la; i++) {
            for (int j = 1; j <= lb; j++) {
                dp[i][j] = a.charAt(i - 1) == b.charAt(j - 1)
                        ? dp[i - 1][j - 1]
                        : 1 + Math.min(dp[i - 1][j - 1],
                        Math.min(dp[i - 1][j], dp[i][j - 1]));
            }
        }
        return dp[la][lb];
    }
}