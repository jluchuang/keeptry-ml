package cn.keeptry.ml.utils

object LogliLikehood {
    def xLogX(x: Long): Double = if (x == 0) 0.0 else x * Math.log(x)

    def entropy(elements: Long*): Double = {
        var sum: Long = 0
        var result: Double = 0.0
        for (element <- elements) {
            result += xLogX(element)
            sum += element
        }
        xLogX(sum) - result
    }

    def logLike(item1Count: Long, item2Count: Long, common: Long, all: Long): Double = {
        val k11 = common // 同时喜欢item1和item2的人数
        val k12 = item1Count - common // 喜欢item2不喜欢item1的人数
        val k21 = item2Count - common // 喜欢item1不喜欢item2的人数
        val k22 = all - item1Count - item2Count + common // 不喜欢item1也不喜欢item2的人数
        val rowEntropy = entropy(k11 + k12, k21 + k22)
        val columnEntropy = entropy(k11 + k21, k12 + k22)
        val matrixEntropy = entropy(k11, k12, k21, k22)
        val sim = Math.max(0.0, 2 * (rowEntropy + columnEntropy - matrixEntropy))

        sim
    }
}
