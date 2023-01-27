from ._corels import audit_sample_robustness
from .generalized_sample_robustness_auditor_ortools_CP import audit_generalized_robustness_or_tools_cp
from .generalized_sample_robustness_auditor_ortools_MIP import audit_robustness_or_tools_MIP
#from .sample_robustness_auditor_ortools_MIP import audit_robustness_or_tools_mip
#import time

class GeneralizedSampleRobustnessAuditor():
    """
    Tool class useful to audit a model's fairness' sample-robustness
    This class only contains one single method for now.
    It has to be instanciated because it is able to use memoisation, and thus possibly avoid useless and costly calls to the solver.

    Parameters
    ----------
    memoisation: int in {0, 1, 2} (default 2)
        Indicates whether to use memoisation or not.
        0 means no memoisation
        1 means simple memoisation (looks if combination of (s_a, s_b, z_a, z_b, epsilon) has already been solved)
        2 means advanced memoisation (same as 1, but also leverages previously solved problems to get objective bounds (lower or upper))
    """
    def __init__(self, memoisation=2):
        self.memo = {}
        self.memoisation = memoisation
        if self.memoisation > 0:
            self.memoRead=0
            self.memoCheck=0
        if self.memoisation == 2:
            self.ub_improvement = 0
            self.lb_improvement = 0

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

        config: int
            Search configuration parameter for the Mistral solver.
            For ORTools, 0 means use the CP solver (no other choice implemented for now)

        debug: int (either 0, 1 or 2)
            Can be used to print useful debug information.
            The higher, the more information printed.

        solver: str (either 'Mistral' or 'OR-tools')
            The CP solver to be used to solve the constrained optimization problem.
            Models for both solvers are provided. 
            Mistral is compiled along with FairCORELS, and can always be used.
            OR-tools has to be installed by the user for this option to be chosen, or else an import error will be raised.

        Returns
        -------
        self : obj
        """
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
        if not isinstance(solver, str):
            raise TypeError("solver must be an str, got: " + str(type(solver)))

        # print args (optional, for debug)
        #print("s_a = ", s_a, ", s_b = ", s_b, ", z_a = ", z_a, ", z_b = ", z_b, ", epsilon = ", epsilon, ", config = ", config, ", debug = ", debug, ", solver = ", solver)

        # (fix) round epsilon, because rounding problems during cast can occur
        epsilon = round(epsilon,4)
        #print(epsilon)

        # init objective bounds
        # not used if -1 and self.memoisation != 2
        objective_lower_bound = -1
        objective_upper_bound = -1

        # if using memoisation, checks whether already solved
        if self.memoisation == 1:
            key = tuple([s_a, s_b, z_a, z_b, epsilon])
            self.memoCheck+=1
            if key in self.memo:
                if debug >= 1:
                    print("Retrieving result saved in memo")
                self.memoRead+=1
                return self.memo[key]
        elif self.memoisation == 2:
            key = tuple([s_a, s_b, z_a, z_b])
            self.memoCheck+=1
            if key in self.memo:
                if epsilon in self.memo[key]: # already solved the exact same problem
                    if debug >= 1:
                        print("Retrieving result saved in memo")
                    self.memoRead+=1
                    return self.memo[key][epsilon]
                else: # already solved the same problem for different value(s) of epsilon
                    for eps in self.memo[key]:
                        old_obj = self.memo[key][eps].score
                        if old_obj == -1: # UNSAT, then UNSAT for every epsilon or else 1.0 always reachable
                            self.memoRead+=1
                            return self.memo[key][eps]
                        if eps < epsilon:
                            if objective_lower_bound == -1 or objective_lower_bound < old_obj:
                                objective_lower_bound = int(old_obj)
                        elif eps > epsilon:
                            if objective_upper_bound == -1 or objective_upper_bound > old_obj:
                                objective_upper_bound = int(old_obj)
                    if objective_lower_bound != -1:
                        self.lb_improvement+=1
                    if objective_upper_bound != -1:
                        self.ub_improvement+=1

        # compute Jaccard-robustness
        if debug > 0:
            print("Objective in [%d, %d]" %(objective_lower_bound, objective_upper_bound))
        if solver == 'Mistral':
            res = audit_generalized_robustness_or_tools_cp(s_a, s_b, z_a, z_b, epsilon, debug, config, objective_lower_bound, objective_upper_bound)
        elif solver == 'OR-tools' and config == 0:
            #display=False
            #start = time.clock()
            res = audit_generalized_robustness_or_tools_cp(s_a, s_b, z_a, z_b, epsilon, debug, objective_lower_bound, objective_upper_bound)
            #dur = time.clock() - start
            #if dur > 1.0:
            #    print("Time = ", dur)
            #    print("s_a = ", s_a, ", s_b = ", s_b, ", z_a = ", z_a, ", z_b = ", z_b, ", epsilon = ", epsilon, ", config = ", config, ", debug = ", debug, ", solver = ", solver)
            #    display=True
        #elif solver == 'OR-tools' and config == 1:
        #    res = audit_robustness_or_tools_mip(s_a, s_b, z_a, z_b, epsilon, debug, objective_lower_bound, objective_upper_bound)
        else:
            print("Unknown solver: ", solver, ", exiting (config was ", config, ").")
            exit()

        # check correctness, prepare return objects
        x_a = res[0]
        y_a = res[1]
        x_b = res[2]
        y_b = res[3]
        x_a_prime = res[4]
        y_a_prime = res[5]
        x_b_prime = res[6]
        y_b_prime = res[7]
        score = res[8]
        status = res[9] # 0 means solved to OPT, 1 means infeasible, 2 means feasible but not OPT
        if status == 0: # found subset violating fairness
            unfairness = abs(((float)(s_a-x_a+x_a_prime) /(float)(z_a-x_a-y_a+x_a_prime+y_a_prime ))  - 	 ((float)(s_b-x_b+x_b_prime) /(float) (z_b-x_b-y_b+x_b_prime+y_b_prime) ))
            if not (unfairness >= epsilon):
                print("Got incorrect result: unfairness is only ", unfairness, " while tolerence was ", epsilon)
            #if not (score == x_a+y_a+x_b+y_b):
            #    print("Got incorrect result from solver: objective is ", score, " != x_a + y_a + x_b + y_b = ", x_a+y_a+x_b+y_b)
            jaccard_dist = 1-(float)((float)((float)(z_a+z_b)-(x_a+x_b+y_a+y_b))/((float)(z_a+z_b)+(x_a_prime+x_b_prime+y_a_prime+y_b_prime)))
        elif status == 1:
            unfairness = abs(((float)(s_a-x_a+x_a_prime) /(float)(z_a-x_a-y_a+x_a_prime+y_a_prime ))  - 	 ((float)(s_b-x_b+x_b_prime) /(float) (z_b-x_b-y_b+x_b_prime+y_b_prime) ))
            if unfairness != 0:
                print("UNSAT problem and unfairness over entire dataset != 0 => Weird case!")
                print("unfairness is ", unfairness)
            jaccard_dist = 1.0
        result = RobustnessAuditResult(x_a, y_a, x_b, y_b, x_a_prime, y_a_prime, x_b_prime, y_b_prime, score, status, unfairness, jaccard_dist)
        
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
        elif self.memoisation == 2:
            print("Advanced memoisation is active.")
            print("     Memo contains %d elements, was checked %d times, avoided %d solver calls."%(len(self.memo), self.memoCheck, self.memoRead))
            print("     Memo improved objective lower bound %d times, objective upper bound %d times."%(self.lb_improvement, self.ub_improvement))
        else:
            print("Memoisation is not active.")


class RobustnessAuditResult():
    """
    Container class for the result of the fairness sample-robustness audit.
    """
    def __init__(self, x_a, y_a, x_b, y_b, x_a_prime, y_a_prime, x_b_prime, y_b_prime, score, status, unfairness, jaccard_dist):
        self.x_a = x_a
        self.x_b = x_b
        self.y_a = y_a
        self.y_b = y_b
        self.x_a_prime = x_a_prime
        self.x_b_prime = x_b_prime
        self.y_a_prime = y_a_prime
        self.y_b_prime = y_b_prime
        self.score = score
        self.status = status
        self.unfairness = unfairness
        self.jaccard_dist = jaccard_dist