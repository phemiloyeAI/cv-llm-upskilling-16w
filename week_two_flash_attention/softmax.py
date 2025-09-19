import math
from typing import List
class SoftMax:
    """
    softmax operation is inefficient since it reads from the vector 3 times:
    - x_max = max(row)
    - normalization_factor, L = e^x1-x_max + e^x2-x_max + ... + e^xn-x_max
    - x1-x_max/L, x2-x_max/L, xn-x_max/L
    """
    def _max(self, vec: List):
        x_max = -math.inf
        for i, x in enumerate(vec):
            x_max = max(x_max, x)
        return x_max
    
    def _norm_factor(self, x_max: float, vec: List):
        l = 0
        for i, x in enumerate(vec):
            l+=math.exp(x-x_max)
        return l 
    
    def _x_term(self, x_max: float, l: float, vec: List):
        probs = []
        for i, x in enumerate(vec):
            norm_x = math.exp(x-x_max) / l
            probs.append(norm_x)
        return probs
    
    def __call__(self, vec: List):
        x_max = self._max(vec)
        l = self._norm_factor(x_max, vec)
        vec_probs = self._x_term(x_max, l, vec)
        
        return vec_probs
    

class OnlineSoftmax:
    """
    use local maxima to normalize x_i. apply a normalization factor e^max_l - max_g to li-1 
    cfactor = e^(max_l - max_g)
    li = l_i-1 * cfactor + e^(xi-max_g)
    """

    def _fused_norm_factor(self, vec: List):
        max_l = -math.inf
        l = 0
        for i, x in enumerate(vec):
            x_max = max(x, max_l)

            if i >= 0 and x_max > max_l:
                # correction factor 
                cfactor = math.exp(max_l - x_max)
                max_l = x_max
                l = l * cfactor + math.exp(x-max_l)

            else:
                l += math.exp(x-x_max)
        return l, max_l
    
    def __call__(self, vec: List):
        l, g_max = self._fused_norm_factor(vec)
        _vec = [math.exp(x-g_max)/l for x in vec]
        return _vec

            
if __name__ == '__main__':
    vec = [3,2,5,1]
    
    softmax = SoftMax()
    online_softmax = OnlineSoftmax()
    assert sum(online_softmax(vec)) == sum(softmax(vec)), 'online softmax and vanila softmax do not give the same results'
    print('Passed!')
    