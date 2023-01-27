#include <mistral_solver.hpp>
#include <mistral_variable.hpp>
#include <mistral_search.hpp>
#include <chrono>

using namespace std;
using namespace Mistral;

class SampleRobustnessAuditor {

private:

	VarArray scope;
	Solver& s;
	Outcome result;
	Goal* goal;
	int s_a, z_a, s_b, z_b;
    int debug;


	//fairness_tolerence is a float in [0,1] that represents the unfairness tolerated.
	//If this value is 0 then the model is 100% fair.
	float epsilon ;


public:
	/*	ub_sp_plus is the number of examples protected with a positive class
		ub_sp_minus is the number of examples protected with a negative class
		ub_su_plus is the number of examples unprotected with a positive class
		ub_su_minus is the number of examples unprotected with a negative class
		TPp is the number of True Positive protected instances (among instances captured by the prefix)
	 */
	SampleRobustnessAuditor(
			int s_a,
			int z_a,
			int s_b,
			int z_b,
			float epsilon,
			Solver& s, 
            int debug,
            int objective_lower_bound,
            int objective_upper_bound
			 ) :
				s_a(s_a),
				z_a(z_a),
				s_b(s_b),
				z_b(z_b),
				epsilon(epsilon),
				s(s),
                debug(debug)
{

        // Original objective bounds
        int obj_lb = 0;
        int obj_ub = z_a+z_b-2;

        // If provided, improved objective bounds
        if(objective_lower_bound != -1){
            obj_lb = objective_lower_bound;
        }
        if(objective_upper_bound != -1){
            obj_ub = objective_upper_bound;
        }


        int oneOverEpsilon = (int)((float)1/epsilon);
		if(debug>=2){
            std::cout << "epsilon= " << epsilon << ", oneOverEpsilon=" << oneOverEpsilon << std::endl;
        }
		//scope[0] is x_a
		scope.add(Variable(0, s_a) );
		//scope[1] is y_a
		scope.add(Variable(0, z_a-s_a) );
		//scope[2] is x_b
		scope.add(Variable(0, s_b) );
		//scope[3] is y_b
		scope.add(Variable(0,z_b-s_b) );
		//scope[4] is (S_a-x_a)*(Z_b-x_b-y_b) 
		scope.add(Variable(0, s_a*z_b) );
		//scope[5] is (S_b-x_b)*(Z_a-x_a-y_a)
		scope.add(Variable(0, s_b*z_a) );
		//scope[6] is (Z_a-x_a-y_a)*(Z_b-x_b-y_b)
		scope.add(Variable(1, z_a*z_b) );
		//scope[7] is objective = x_a + x_b + y_a + y_b
		scope.add(Variable(obj_lb, obj_ub) );
        // below: to force assignation of #examples removed
		/*s.add(scope[0] == 650);
		s.add(scope[1] == 0);
		s.add(scope[2] == 0);
		s.add(scope[3] == 0);*/
		result =UNSAT;
		
		int constant ;

		// (S_a-x_a)*(Z_b-x_b-y_b)
		s.add(scope[4] == (-scope[0] + s_a)*(-scope[2] -scope[3] + z_b));
		// (S_b-x_b)*(Z_a-x_a-y_a)
		s.add(scope[5] == (-scope[2] + s_b)*(-scope[0] -scope[1] + z_a));

		// (Z_a-x_a-y_a)*(Z_b-x_b-y_b)
		s.add(scope[6] ==(-scope[0] -scope[1] + z_a)*(-scope[2] -scope[3] + z_b));

		// Enforce that fairness is violated -> Faster without the Abs constraint!
		s.add( (scope[6] < (scope[4] - scope[5])*oneOverEpsilon) or (scope[6] < (scope[5] - scope[4])*oneOverEpsilon));

		//objective
		s.add(scope[7] == (scope[0]+scope[1]+scope[2]+scope[3]));
		
		goal = new Goal(Goal::MINIMIZATION,scope[7]);//Sum(removed_examples, objective_coefficients)
		//std::cout << "s.objective:" << goal <<  std::endl;
}

	void run(int verbosity, int config){
		s.parameters.verbosity = verbosity;
		//s.parameters.time_limit = t_out;
		//std::cout <<  s << std::endl;
		//s.rewrite();
		s.consolidate();
		s.set_goal(goal);
        if(debug>=2){
            std::cout << "c solver internal  representation " << s << std::endl;
        }
		//std::cout << "goal assignated" <<  std::endl;
		if (!config)
			//use default solver strategy
			result =  s.solve();
		else
		{
			bool branch_on_decision_only = true;
			RestartPolicy *_option_policy = NULL;
			BranchingHeuristic *_option_heuristic ;


			switch(config) {
		    case -1:
				//std::cout << "Optimisation test policy" << std::endl;
				branch_on_decision_only= true;
				_option_policy = new Luby();
				_option_policy->base = 128;
				_option_heuristic = new LastConflict < GenericDVO < MinDomainOverWeight, 1, ConflictCountManager >, SolutionGuided< MinValue, RandomMinMax >, SolutionGuided< MinValue, RandomMinMax >, 1 > (&s);

			case 1 :
				branch_on_decision_only= true;
				//This is used in the competition
				_option_policy = new Luby();
				_option_heuristic = new LastConflict < GenericDVO < MinDomainOverWeight, 2, ConflictCountManager >,  Guided< MinValue >,  Guided< MinValue >, 1 > (&s);
				break;
			case 2 :
				branch_on_decision_only= true;
				_option_policy = new Luby();
				_option_heuristic = new LastConflict < GenericDVO < MinDomainOverWeight, 2, ConflictCountManager >,  Guided< RandomMinMax >,  Guided< RandomMinMax >, 1 > (&s);
				break;

			case 3 :
				branch_on_decision_only= true;

				_option_policy = new Luby();
				_option_heuristic = new LastConflict < GenericDVO < MinDomainOverWeight, 2, ConflictCountManager >,  Guided< MiddleValue >,  Guided< MiddleValue >, 1 > (&s);
				break;

			case 4 :
				branch_on_decision_only= true;
				_option_policy = new Geometric();
				_option_heuristic = new LastConflict < GenericDVO < MinDomainOverWeight, 2, ConflictCountManager >,  Guided< MinValue >,  Guided< MinValue >, 1 > (&s);
				break;
			case 5 :
				branch_on_decision_only= true;
				_option_policy = new Geometric();
				_option_heuristic = new LastConflict < GenericDVO < MinDomainOverWeight, 2, ConflictCountManager >,  Guided< RandomMinMax >,  Guided< RandomMinMax >, 1 > (&s);
				break;

			case 6 :
				branch_on_decision_only= true;
				_option_policy = new Geometric();
				_option_heuristic = new LastConflict < GenericDVO < MinDomainOverWeight, 2, ConflictCountManager >,  Guided< MiddleValue >,  Guided< MiddleValue >, 1 > (&s);
				break;


			case 7 :
				branch_on_decision_only= true;
				_option_policy = new NoRestart();
				_option_heuristic = new LastConflict < GenericDVO < MinDomainOverWeight, 2, ConflictCountManager >,  Guided< MinValue >,  Guided< MinValue >, 1 > (&s);
				break;
			case 8 :
				branch_on_decision_only= true;
				_option_policy = new NoRestart();
				_option_heuristic = new LastConflict < GenericDVO < MinDomainOverWeight, 2, ConflictCountManager >,  Guided< RandomMinMax >,  Guided< RandomMinMax >, 1 > (&s);
				break;

			case 9 :
				branch_on_decision_only= true;
				_option_policy = new NoRestart();
				_option_heuristic = new LastConflict < GenericDVO < MinDomainOverWeight, 2, ConflictCountManager >,  Guided< MiddleValue >,  Guided< MiddleValue >, 1 > (&s);
				break;

			case 10 :
				branch_on_decision_only= false;
				//This is used in the competition
				_option_policy = new Luby();
				_option_heuristic = new LastConflict < GenericDVO < MinDomainOverWeight, 2, ConflictCountManager >,  Guided< MinValue >,  Guided< MinValue >, 1 > (&s);
				break;
			case 11 :
				branch_on_decision_only= false;
				_option_policy = new Luby();
				_option_heuristic = new LastConflict < GenericDVO < MinDomainOverWeight, 2, ConflictCountManager >,  Guided< RandomMinMax >,  Guided< RandomMinMax >, 1 > (&s);
				break;

			case 12 :
				branch_on_decision_only= false;

				_option_policy = new Luby();
				_option_heuristic = new LastConflict < GenericDVO < MinDomainOverWeight, 2, ConflictCountManager >,  Guided< MiddleValue >,  Guided< MiddleValue >, 1 > (&s);
				break;

			case 13 :
				branch_on_decision_only= false;
				_option_policy = new Geometric();
				_option_heuristic = new LastConflict < GenericDVO < MinDomainOverWeight, 2, ConflictCountManager >,  Guided< MinValue >,  Guided< MinValue >, 1 > (&s);
				break;
			case 14 :
				branch_on_decision_only= false;
				_option_policy = new Geometric();
				_option_heuristic = new LastConflict < GenericDVO < MinDomainOverWeight, 2, ConflictCountManager >,  Guided< RandomMinMax >,  Guided< RandomMinMax >, 1 > (&s);
				break;

			case 15 :
				branch_on_decision_only= false;
				_option_policy = new Geometric();
				_option_heuristic = new LastConflict < GenericDVO < MinDomainOverWeight, 2, ConflictCountManager >,  Guided< MiddleValue >,  Guided< MiddleValue >, 1 > (&s);
				break;


			case 16 :
				branch_on_decision_only= false;
				_option_policy = new NoRestart();
				_option_heuristic = new LastConflict < GenericDVO < MinDomainOverWeight, 2, ConflictCountManager >,  Guided< MinValue >,  Guided< MinValue >, 1 > (&s);
				break;
			case 17 :
				branch_on_decision_only= false;
				_option_policy = new NoRestart();
				_option_heuristic = new LastConflict < GenericDVO < MinDomainOverWeight, 2, ConflictCountManager >,  Guided< RandomMinMax >,  Guided< RandomMinMax >, 1 > (&s);
				break;

			case 18 :
				branch_on_decision_only= false;
				_option_policy = new NoRestart();
				_option_heuristic = new LastConflict < GenericDVO < MinDomainOverWeight, 2, ConflictCountManager >,  Guided< MiddleValue >,  Guided< MiddleValue >, 1 > (&s);
				break;
			default :
				std::cout << " c confid not used " << config << std::endl;

			}


			s.parameters.activity_decay = 0.95;
			s.parameters.seed = 10 ;
			if (branch_on_decision_only){
				//This one is used only when branching on decision variables
				VarArray decision_variables;
				decision_variables.add (scope[0]) ;
				decision_variables.add (scope[1]) ;
				decision_variables.add (scope[2]) ;
				decision_variables.add (scope[3]) ;

				result = s.depth_first_search(decision_variables, _option_heuristic, _option_policy);
			}
			else
				result = s.depth_first_search(s.variables , _option_heuristic, _option_policy);

		}
	}

	void print_statistics(){
		s.statistics.print_full(std::cout);
	}


	void print_and_verify_solution(){

		if (result )
		{
			std::cout <<  " \n \n c Solution Found! "  << 	std::endl;

			//for ( unsigned int i= 0 ; i< scope.size ; ++i)
			//	std::cout <<  " c Solution value of scope[" << i <<  "]  is " << scope[i].get_solution_int_value() << std::endl;

			// decision vars
			int x_a= scope[0].get_solution_int_value();
			int y_a= scope[1].get_solution_int_value();
			int x_b= scope[2].get_solution_int_value();
			int y_b= scope[3].get_solution_int_value();

			// other vars
			int prod1 = scope[4].get_solution_int_value();
			int prod2 = scope[5].get_solution_int_value();
			int otherterm = scope[6].get_solution_int_value();

			// decision vars
			std::cout <<  " x_a " <<  x_a << std::endl;
			std::cout <<  " y_a " <<  y_a << std::endl;
			std::cout <<  " x_b " <<  x_b << std::endl;
			std::cout <<  " y_b  " <<  y_b << std::endl;

			// other vars	

			//std::cout <<  " prod1  " <<  prod1 << std::endl;
			//std::cout <<  " prod2  " <<  prod2 << std::endl;
			//std::cout <<  " otherterm  " <<  otherterm << std::endl;


			std::cout <<  " fairness is violated (for the opt-CSP): " << (abs(prod1-prod2) >= epsilon*otherterm)  << std::endl;
			float fairness = ((float)(s_a-x_a) /(float)(z_a-x_a-y_a ))  - 	 ((float)(s_b-x_b) /(float) (z_b-x_b-y_b) );
			std::cout <<  " a-ratio =" << ((float)(s_a-x_a) /(float) (z_a-x_a-y_a )) << ", b-ratio = " << ((float)(s_b-x_b) /(float)(z_b-x_b-y_b)) << std::endl;
			std::cout <<  " c Unfairness over found subset (as float) is "  << fairness << " expected " << epsilon << ")" <<  std::endl;
			std::cout <<  " c Subset is at Jaccard dist (~fairness sample robustness) = "  << (float)((float)(x_a+x_b+y_a+y_b)/(float)(z_a+z_b)) << std::endl; // assume z_a+z_b = total number of samples in dataset!

			//int discrete_fairness= (nb_unprotected * (sp_plus+sp_minus)) - (nb_protected * (su_plus +su_minus) )  ;
			//std::cout <<  " c Fairness (discrete constraint) is "  << discrete_fairness << std::endl;
			std::cout <<  " c Solution Verified"  << 	std::endl;

		}
		else
			std::cout <<  " c No Solution! "  << 	std::endl;
	}

    void fill_solution(int * res){
        if (result )
		{
			// decision vars
			int x_a= scope[0].get_solution_int_value();
			int y_a= scope[1].get_solution_int_value();
			int x_b= scope[2].get_solution_int_value();
			int y_b= scope[3].get_solution_int_value();
            //objective
			int objective = scope[7].get_solution_int_value();
            res[0]=x_a;
            res[1]=y_a;
            res[2]=x_b;
            res[3]=y_b;
            res[4]=objective; // redundant, only to check
            res[5]=0;
		}
		else {
            res[0]=-1;
            res[1]=-1;
            res[2]=-1;
            res[3]=-1;
            res[4]=-1; // redundant, only to check
			res[5]=1;
        }
    }


    int get_objective(){
        if (result)
        return scope[7].get_solution_int_value();
        else
        return -1;
    }
};

int quantify_robustness(int s_a, int s_b, int z_a, int z_b, float epsilon, int config, int debug, int objective_lower_bound, int objective_upper_bound, int* res){
	if(debug > 0){
        std::cout <<  " c Instance data :  " <<  	std::endl;

        std::cout <<  " c s_a "  << s_a <<	std::endl;
        std::cout <<  " c z_a "  << z_a <<	std::endl;
        std::cout <<  " c s_b "  << s_b <<	std::endl;
        std::cout <<  " c z_b "  << z_b <<	std::endl;

        std::cout <<  " c Unfairness tolerance (should be in [0,1]): "  << epsilon << 	std::endl;
        std::cout <<  " c Unfairness over dataset (should be in [0,1]): "  << (float) fabs((float) ((float)s_a/(float)z_a) - (float) ((float)s_b/(float)z_b)) << 	std::endl;
        std::cout <<  " c Solver configuration :  " <<  config << std::endl;
    }

    bool ok = false;

    while(not ok){
        try{
            ok = true;
            Solver solver;

            //std::cout <<  " c run the solver .. and wait for magic  " <<  	std::endl;
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

            SampleRobustnessAuditor audit_sample_robustness(s_a, z_a, s_b, z_b, epsilon, solver, debug, objective_lower_bound, objective_upper_bound);

            audit_sample_robustness.run(0 , config);
            //}
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            if(debug > 0){
                audit_sample_robustness.print_statistics();
                audit_sample_robustness.print_and_verify_solution();
            }
            audit_sample_robustness.fill_solution(res);
            if(debug > 0){
                std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
            }
            return audit_sample_robustness.get_objective();
    } catch(ExceptionConstraintInconsistent& e){
        std::cout << "An exception occured, relaunching search with other configuration : " << config << " -> " << ++config << std::endl;
        ok = false;
    }
    }
    
}



