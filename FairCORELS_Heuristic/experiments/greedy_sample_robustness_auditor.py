import sys

def unf(s_a, x_a, s_b, x_b):
    return abs( (s_a/x_a) - (s_b/x_b) )

def audit_sample_robustness_greedy_with_manual_calc(s_a, s_b, x_a, x_b, epsilon, debug, accu=0):
    # check first possibility: 
    # remove example from group a
    # not satisfying the measure
    if x_a > 1:
        firstMoveImprovement = s_a / ((x_a - 1) * x_a)
    else:
        firstMoveImprovement = -1

    if debug:
        if x_a > 1:
            firstMoveImprovementCheck = unf(s_a, x_a - 1, s_b, x_b) - unf(s_a, x_a, s_b, x_b)
            if round(firstMoveImprovementCheck,10) != round(firstMoveImprovement,10):
                print("error in check 1: expected %f, got %f" %(firstMoveImprovement, firstMoveImprovementCheck))
    # check second possibility: 
    # remove example from group b
    # satisfying the measure
    if s_b > 0 and x_b > 1:
        secondMoveImprovement = (x_b - s_b) / ((x_b - 1) * x_b)
    else:
        secondMoveImprovement = -1

    if debug:
        if s_b > 0 and x_b > 1:
            secondMoveImprovementCheck = unf(s_a, x_a, s_b - 1, x_b - 1) - unf(s_a, x_a, s_b, x_b)
            if round(secondMoveImprovementCheck, 10) != round(secondMoveImprovement, 10):
                print("error in check 2: expected %f, got %f" %(secondMoveImprovement, secondMoveImprovementCheck))

    # check third possibility: 
    # remove example from group a
    # satisfying the measure
    if s_a > 0 and x_a > 1:
        if ((s_a - 1) / (x_a - 1)) < (s_b/x_b):
            thirdMoveImprovement = 2*(s_b/x_b) - ( ( (2*s_a*x_a) - x_a - s_a ) / (x_a * (x_a - 1))   )
        else:
            thirdMoveImprovement = 0
    else:
        thirdMoveImprovement = -1

    if debug:
        if s_a > 0 and x_a > 1:
            thirdMoveImprovementCheck = unf(s_a-1, x_a-1, s_b, x_b) - unf(s_a, x_a, s_b, x_b)
            if thirdMoveImprovementCheck < 0:
                thirdMoveImprovementCheck = 0
            if round(thirdMoveImprovementCheck, 10) != round(thirdMoveImprovement, 10):
                print("error in check 3: expected %f, got %f" %(thirdMoveImprovement, thirdMoveImprovementCheck))

    # check fourth possibility: 
    # remove example from group b
    # not satisfying the measure
    if x_b > 1:
        if ((s_a ) / (x_a )) < (s_b/(x_b-1)):
            fourthMoveImprovement = ( ((2*s_b*x_b) - s_b) / (x_b * (x_b - 1)) ) - ((2*s_a)/x_a   )
        else:
            fourthMoveImprovement = 0
    else:
        fourthMoveImprovement = -1

    if debug:
        if x_a > 1:
            fourthMoveImprovementCheck = unf(s_a, x_a, s_b, x_b-1) - unf(s_a, x_a, s_b, x_b)
            if fourthMoveImprovementCheck < 0:
                fourthMoveImprovementCheck = 0
            if round(fourthMoveImprovementCheck, 10) != round(fourthMoveImprovement, 10):
                print("error in check 4: expected %f, got %f" %(fourthMoveImprovement, fourthMoveImprovementCheck))

    improvementsList = [firstMoveImprovement, secondMoveImprovement, thirdMoveImprovement, fourthMoveImprovement]

    if max(improvementsList) <= 0:
        return -1, -1, -1, -1, (s_a+s_b+z_a+z_b)+accu, 1

    bestMove = improvementsList.index(max(improvementsList))
    if bestMove == 0:
        if unf(s_a, x_a - 1, s_b, x_b) > epsilon:
            return s_a, x_a, s_b, x_b, 1+accu, 0
        else:
            return audit_sample_robustness_greedy(s_a, s_b, x_a - 1, x_b, epsilon, debug, accu = accu + 1)
    elif bestMove == 1:
        if unf(s_a, x_a, s_b-1, x_b-1) > epsilon:
            return s_a, x_a, s_b-1, x_b-1, 1+accu, 0
        else:
            return audit_sample_robustness_greedy(s_a, s_b-1, x_a, x_b-1, epsilon, debug, accu = accu + 1)
    elif bestMove == 2:
        if unf(s_a-1, x_a-1, s_b, x_b) > epsilon:
            return s_a-1, x_a-1, s_b, x_b, 1+accu, 0
        else:
            return audit_sample_robustness_greedy(s_a-1, s_b, x_a-1, x_b, epsilon, debug, accu = accu + 1)      
    elif bestMove == 3:
        if unf(s_a, x_a, s_b, x_b-1) > epsilon:
            return s_a, x_a, s_b, x_b-1, 1+accu, 0
        else:
            return audit_sample_robustness_greedy(s_a, s_b, x_a, x_b - 1, epsilon, debug, accu = accu + 1)

def audit_sample_robustness_greedy(s_a, s_b, x_a, x_b, epsilon, debug, accu=0):
    # check first possibility: 
    # remove example from group a
    # not satisfying the measure
    if x_a > 1:
        firstMoveImprovement = unf(s_a, x_a - 1, s_b, x_b) - unf(s_a, x_a, s_b, x_b)
    else:
        firstMoveImprovement = -1

    # check second possibility: 
    # remove example from group b
    # satisfying the measure
    if s_b > 0 and x_b > 1:
        secondMoveImprovement = unf(s_a, x_a, s_b - 1, x_b - 1) - unf(s_a, x_a, s_b, x_b)
    else:
        secondMoveImprovement = -1

    # check third possibility: 
    # remove example from group a
    # satisfying the measure
    if s_a > 0 and x_a > 1:
        thirdMoveImprovement = unf(s_a-1, x_a-1, s_b, x_b) - unf(s_a, x_a, s_b, x_b)
    else:
        thirdMoveImprovement = -1

    # check fourth possibility: 
    # remove example from group b
    # not satisfying the measure
    if x_b > 1:
        fourthMoveImprovement = unf(s_a, x_a, s_b, x_b-1) - unf(s_a, x_a, s_b, x_b)
    else:
        fourthMoveImprovement = -1

    improvementsList = [firstMoveImprovement, secondMoveImprovement, thirdMoveImprovement, fourthMoveImprovement]

    if max(improvementsList) <= 0:
        return -1, -1, -1, -1, (s_a+s_b+x_a+x_b)+accu, 1

    bestMove = improvementsList.index(max(improvementsList))
    if bestMove == 0:
        if unf(s_a, x_a - 1, s_b, x_b) > epsilon:
            return s_a, x_a-1, s_b, x_b, 1+accu, 0
        else:
            return audit_sample_robustness_greedy(s_a, s_b, x_a - 1, x_b, epsilon, debug, accu = accu + 1)
    elif bestMove == 1:
        if unf(s_a, x_a, s_b-1, x_b-1) > epsilon:
            return s_a, x_a, s_b-1, x_b-1, 1+accu, 0
        else:
            return audit_sample_robustness_greedy(s_a, s_b-1, x_a, x_b-1, epsilon, debug, accu = accu + 1)
    elif bestMove == 2:
        if unf(s_a-1, x_a-1, s_b, x_b) > epsilon:
            return s_a-1, x_a-1, s_b, x_b, 1+accu, 0
        else:
            return audit_sample_robustness_greedy(s_a-1, s_b, x_a-1, x_b, epsilon, debug, accu = accu + 1)      
    elif bestMove == 3:
        if unf(s_a, x_a, s_b, x_b-1) > epsilon:
            return s_a, x_a, s_b, x_b-1, 1+accu, 0
        else:
            return audit_sample_robustness_greedy(s_a, s_b, x_a, x_b - 1, epsilon, debug, accu = accu + 1)

def audit_sample_robustness_greedy_iterative(s_a_arg, s_b_arg, x_a_arg, x_b_arg, epsilon, debug, accu=0):
    s_a = s_a_arg
    s_b = s_b_arg
    x_a = x_a_arg
    x_b = x_b_arg

    accu = 0
    while unf(s_a, x_a, s_b, x_b) <= epsilon:    
        # check first possibility: 
        # remove example from group a
        # not satisfying the measure
        if x_a > 1 and s_a < x_a:
            firstMoveImprovement = unf(s_a, x_a - 1, s_b, x_b) - unf(s_a, x_a, s_b, x_b)
        else:
            firstMoveImprovement = -1

        # check second possibility: 
        # remove example from group b
        # satisfying the measure
        if s_b > 0 and x_b > 1:
            secondMoveImprovement = unf(s_a, x_a, s_b - 1, x_b - 1) - unf(s_a, x_a, s_b, x_b)
        else:
            secondMoveImprovement = -1

        # check third possibility: 
        # remove example from group a
        # satisfying the measure
        if s_a > 0 and x_a > 1:
            thirdMoveImprovement = unf(s_a-1, x_a-1, s_b, x_b) - unf(s_a, x_a, s_b, x_b)
        else:
            thirdMoveImprovement = -1

        # check fourth possibility: 
        # remove example from group b
        # not satisfying the measure
        if x_b > 1 and s_b < x_b:
            fourthMoveImprovement = unf(s_a, x_a, s_b, x_b-1) - unf(s_a, x_a, s_b, x_b)
        else:
            fourthMoveImprovement = -1

        improvementsList = [firstMoveImprovement, secondMoveImprovement, thirdMoveImprovement, fourthMoveImprovement]

        if max(improvementsList) <= 0:
            return -1, -1, -1, -1, (s_a_arg+s_b_arg+x_a_arg+x_b_arg), 1

        bestMove = improvementsList.index(max(improvementsList))
        if bestMove == 0:
            x_a -= 1
            accu += 1
        elif bestMove == 1:
            s_b -= 1
            x_b -= 1
            accu += 1
        elif bestMove == 2:
            s_a -= 1
            x_a -= 1
            accu += 1
        elif bestMove == 3:
            x_b -= 1
            accu += 1

    return s_a, x_a, s_b, x_b, accu, 0
    

class GreedySampleRobustnessAuditor():
    """
    Tool class useful to audit a model's fairness' sample-robustness
    This class only contains one single method for now.
    It has to be instanciated because it is able to use memoisation, and thus possibly avoid useless and costly calls to the greedy algorithm.

    Parameters
    ----------
    memoisation: int in {0, 1, 2} (default 2)
        Indicates whether to use memoisation or not.
        0 means no memoisation
        1 means simple memoisation (looks if combination of (s_a, s_b, z_a, z_b, epsilon) has already been solved)
    """
    def __init__(self, memoisation=1):
        self.memo = {}
        self.memoisation = memoisation
        if self.memoisation > 0:
            self.memoRead=0
            self.memoCheck=0

    def audit_robustness(self, s_a, s_b, z_a, z_b, epsilon, config=0, debug=0, solver='Mistral'):
        """
        Computes sample-robustness of a model's fairness.

        Parameters
        ----------
        s_a, s_b, z_a, z_b: int
            These paremeters correspond to protected groups a and b measurements for the fairness metric at hand.
            Unfairness is expressed as:
            abs((s_a/z_a) - (s_b/z_b)) 

        epsilon: float
            Unfairness tolerence to be met on the entire set.

        debug: int (either 0, 1 or 2)
            Can be used to print useful debug information.
            The higher, the more information printed.

        Returns
        -------
        self : obj
        """
        sys.setrecursionlimit(s_a+s_b+z_a+z_b)
        #print("params: ", s_a, s_b, z_a, z_b)
        # check args
        if not isinstance(s_a, int):
            raise TypeError("s_a must be an int, got: " + str(type(s_a)))
        if not isinstance(s_b, int):
            raise TypeError("s_b must be an int, got: " + str(type(s_b)))
        if not isinstance(z_a, int):
            raise TypeError("z_a must be an int, got: " + str(type(z_a)))
        if not isinstance(z_b, int):
            raise TypeError("z_b must be an int, got: " + str(type(z_b)))
        if not isinstance(epsilon, float):
            raise TypeError("epsilon must be a float, got: " + str(type(epsilon)))
        if not isinstance(config, int):
            raise TypeError("config must be an int, got: " + str(type(config)))
        if z_a <= 0:
            raise ValueError("z_a should be strictly greater than 0, got : z_a = %d. Exiting." %(z_a))
        if s_a < 0:
            raise ValueError("s_a should be greater or equal to 0, got : s_a = %d. Exiting." %(s_a))
        if z_b <= 0:
            raise ValueError("z_b should be strictly greater than 0, got : z_b = %d. Exiting." %(z_b))
        if s_b < 0:
            raise ValueError("s_b should be greater or equal to 0, got : s_b = %d. Exiting." %(s_b))
        if s_a > z_a:
            raise ValueError("s_a should be no greater than z_a, got : s_a = %d, z_a = %d. Exiting." %(s_a, z_a))
        if s_b > z_b:
            raise ValueError("s_b should be no greater than z_b, got : s_b = %d, z_b = %d. Exiting." %(s_b, z_b))
        if not isinstance(debug, int):
            raise TypeError("debug must be an int, got: " + str(type(debug)))
        if not(debug == 0 or debug == 1 or debug == 2):
            raise ValueError("debug should be either 0, 1 or 2, got : ", debug, ". Exiting.")

        epsilon=round(epsilon,5)
        # if using memoisation, checks whether already solved
        if self.memoisation == 1:
            key = tuple([s_a, s_b, z_a, z_b, epsilon])
            self.memoCheck+=1
            if key in self.memo:
                if debug >= 1:
                    print("Retrieving result saved in memo")
                self.memoRead+=1
                return self.memo[key]

        # compute Jaccard-robustness
        res = audit_sample_robustness_greedy_iterative(s_a, s_b, z_a, z_b, epsilon, debug) #audit_sample_robustness_greedy_with_manual_calc
        #print(res)
        # check correctness, prepare return objects
        x_a = s_a-res[0]
        y_a = z_a-res[1]-x_a
        x_b = s_b-res[2]
        y_b = z_b-res[3]-x_b
        score = res[4]
        status = res[5] # 0 means solved to OPT, 1 means infeasible, 2 means feasible but not OPT
        #print(x_a, y_a, x_b, y_b)
        if (s_b-x_b) >  (z_b-x_b-y_b):
            print("error, exiting (b)")
        if (s_a-x_a) >  (z_a-x_a-y_a ):
            print("error, exiting (a)")
        if status == 0: # found subset violating fairness
            unfairness = abs(((s_a-x_a) /(z_a-x_a-y_a ))  - 	 ((s_b-x_b) / (z_b-x_b-y_b) ))
            #print(s_a-x_a, "/", z_a-x_a-y_a , "-", s_b-x_b, "/",z_b-x_b-y_b,"=" ,unfairness)
            if not (unfairness >= epsilon):
                print("Got incorrect result: unfairness is only ", unfairness, " while tolerence was ", epsilon)
            if not (score == x_a+y_a+x_b+y_b):
                print("Got incorrect result from solver: objective is ", score, " != x_a + y_a + x_b + y_b = ", x_a+y_a+x_b+y_b)
            jaccard_dist = (float)((float)(x_a+x_b+y_a+y_b)/(float)(z_a+z_b))
        elif status == 1:
            unfairness = abs(((float)(s_a) /(float)(z_a))  - 	 ((float)(s_b) /(float) (z_b) ))
            if unfairness != 0:
                print("UNSAT problem and unfairness over entire dataset != 0 => Weird case!")
                print("unfairness is ", unfairness)
            jaccard_dist = 1.0
        result = RobustnessAuditResult(x_a, y_a, x_b, y_b, score, status, unfairness, jaccard_dist)
        
        # print result
        #if display:
        #    print("Result : ", " x_a = ", x_a, ", y_a = ", y_a, ", x_b = ", x_b, ", y_b = ", y_b, ",score = ", score, ", status = ", status, ", unfairness = ", unfairness, ", jaccard_dist = ", jaccard_dist)

        # if using memoisation, saves result in memo
        if self.memoisation == 1:
            key = tuple([s_a, s_b, z_a, z_b, epsilon])
            self.memo[key] = result
        elif self.memoisation == 2:
            key = tuple([s_a, s_b, z_a, z_b])
            if not key in self.memo:
                self.memo[key] = {}
            self.memo[key][epsilon] = result

        return result

    """
    prints useful statistics about the use of memoisation
    Can be used to estimate the memory use and eventual gain induced by the use of the memoisation technique
    """
    def print_memo_statistics(self):
        if self.memoisation == 1:
            print("Simple memoisation is active.")
            print("     Memo contains %d elements, was checked %d times, avoided %d solver calls."%(len(self.memo), self.memoCheck, self.memoRead))
        else:
            print("Memoisation is not active.")


class RobustnessAuditResult():
    """
    Container class for the result of the fairness sample-robustness audit.
    """
    def __init__(self, x_a, y_a, x_b, y_b, score, status, unfairness, jaccard_dist):
        self.x_a = x_a
        self.x_b = x_b
        self.y_a = y_a
        self.y_b = y_b
        self.score = score
        self.status = status
        self.unfairness = unfairness
        self.jaccard_dist = jaccard_dist


def print_audit(audit):
    print("x_a=",audit.x_a)
    print("x_b=",audit.x_b)
    print("y_a=",audit.y_a)
    print("y_b=",audit.y_b)
    print("score=",audit.score)
    print("unfairness (saved)=", audit.unfairness)
    print("jaccard_dist=",audit.jaccard_dist)



"""auditor = GreedySampleRobustnessAuditor()
audit1 = auditor.audit_robustness(616, 486, 11454, 7203, 1-0.91, debug=1)
print_audit(audit1)
from faircorels import SampleRobustnessAuditor
auditor_e = SampleRobustnessAuditor()
audit2 = auditor_e.audit_robustness(616, 486, 11454, 7203, 1-0.91, solver='OR-tools')
print_audit(audit2)"""
