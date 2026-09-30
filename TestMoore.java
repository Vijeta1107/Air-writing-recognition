import java.util.ArrayList;
import java.util.List;

public class TestMoore {

    static class Point { int x, y; Point(int x, int y){this.x=x;this.y=y;} }

    public static void main(String[] args) {
        int w = 10, h = 10;
        int[] pixels = new int[w*h];
        // Draw a 3x3 square at (3,3)
        for(int r=3; r<=5; r++){
            for(int c=3; c<=5; c++){
                pixels[r*w+c] = 255;
            }
        }

        List<Point> b = extractContour(pixels, w, h);
        System.out.println("Boundary size: " + b.size());
        for(Point p : b){
            System.out.println(p.x + "," + p.y);
        }
    }

    private static List<Point> extractContour(int[] pixels, int w, int h) {
        List<Point> boundary = new ArrayList<>();
        int startX = -1, startY = -1;
        outer:
        for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {
                if (pixels[row * w + col] > 128) {
                    startX = col; startY = row;
                    break outer;
                }
            }
        }
        if (startX == -1) return boundary;

        int[] dx = { 0, 1, 1, 1, 0, -1, -1, -1 };
        int[] dy = { -1, -1, 0, 1, 1, 1, 0, -1 };

        int currX = startX;
        int currY = startY;
        int backDir = 6; // West

        do {
            boundary.add(new Point(currX, currY));
            // FIXED MOORE NEIGHBORHOOD: Start from backDir + 1
            int nextDir = (backDir + 1) % 8;
            int foundDir = -1;
            for (int i = 0; i < 8; i++) {
                int dir = (nextDir + i) % 8;
                int nx = currX + dx[dir];
                int ny = currY + dy[dir];
                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                    if (pixels[ny * w + nx] > 128) {
                        foundDir = dir;
                        break;
                    }
                }
            }
            if (foundDir == -1) break;
            currX += dx[foundDir];
            currY += dy[foundDir];
            backDir = (foundDir + 4) % 8;

            if (boundary.size() > 100) break;
        } while (currX != startX || currY != startY);

        return boundary;
    }
}
