package Text;
public class Number {
    public void arrayAdds(int[][] a) {
        int sum=0;
        for(int row = 0;row < 100;row++)
            for(int col = 0;col < 5;col++)
                sum = sum + a[row][col];
        for(int row = 0;row < 100;row++)
            for(int col = 0;col < 5;col++)
                sum = sum + a[row][col];
    }
}