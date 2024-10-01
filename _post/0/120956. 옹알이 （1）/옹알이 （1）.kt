class Solution {
    fun solution(babbling: Array<String>): Int {
        var answer: Int = 0
        var babblingList=arrayOf("aya","ye","woo","ma")
        
        
        for (item in babbling) {
            var str:String = item
            for (babblingCheck in babblingList){
                if(str.indexOf(babblingCheck)!=-1)
                 str = str.replace(babblingCheck, "!")
            }
            while(str.indexOf("!")!=-1){
                
                 str = str.replace("!", "")
            }
            if(str=="")
                answer++;
        }    
        
        return answer
    }
}