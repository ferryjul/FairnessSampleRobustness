#include "ortools/sat/cp_model.h"
#include "ortools/sat/model.h"
#include "ortools/sat/sat_parameters.pb.h"

namespace operations_research {
namespace sat {

void AuditSampleFairness(int s_a,
			int z_a,
			int s_b,
			int z_b,
            float epsilon,
            int objective_lower_bound,
            int objective_upper_bound) {

  CpModelBuilder cp_model;

  int oneOverEpsilon = (int) ((float)1/epsilon);
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

  const Domain domain_x_a(0, s_a);
  const IntVar x_a = cp_model.NewIntVar(domain_x_a).WithName("x_a");

  const Domain domain_x_b(0, s_b);
  const IntVar x_b = cp_model.NewIntVar(domain_x_b).WithName("x_b");

  const Domain domain_y_a(0, z_a - s_a);
  const IntVar y_a = cp_model.NewIntVar(domain_y_a).WithName("y_a");

  const Domain domain_y_b(0, z_b - s_b);
  const IntVar y_b = cp_model.NewIntVar(domain_y_b).WithName("y_b");

  const Domain domain_s_a_minus_x_a(0, s_a);
  const IntVar s_a_minus_x_a = cp_model.NewIntVar(domain_s_a_minus_x_a).WithName("s_a_minus_x_a");
  LinearExpr s_a_minus_x_a_expr;
  s_a_minus_x_a_expr.AddTerm(x_a, -1);
  s_a_minus_x_a_expr.AddConstant(s_a);
  cp_model.AddEquality(s_a_minus_x_a, s_a_minus_x_a_expr);

  const Domain domain_s_b_minus_x_b(0, s_b);
  const IntVar s_b_minus_x_b = cp_model.NewIntVar(domain_s_b_minus_x_b).WithName("s_b_minus_x_b");
  LinearExpr s_b_minus_x_b_expr;
  s_b_minus_x_b_expr.AddTerm(x_b, -1);
  s_b_minus_x_b_expr.AddConstant(s_b);
  cp_model.AddEquality(s_b_minus_x_b, s_b_minus_x_b_expr);

  const Domain domain_z_a_minus_x_a_minus_y_a(1, z_a);
  const IntVar z_a_minus_x_a_minus_y_a = cp_model.NewIntVar(domain_z_a_minus_x_a_minus_y_a).WithName("z_a_minus_x_a_minus_y_a");
  LinearExpr z_a_minus_x_a_minus_y_a_expr;
  z_a_minus_x_a_minus_y_a_expr.AddTerm(x_a, -1);
  z_a_minus_x_a_minus_y_a_expr.AddTerm(y_a, -1);
  z_a_minus_x_a_minus_y_a_expr.AddConstant(z_a);
  cp_model.AddEquality(z_a_minus_x_a_minus_y_a, z_a_minus_x_a_minus_y_a_expr);

  const Domain domain_z_b_minus_x_b_minus_y_b(1, z_b);
  const IntVar z_b_minus_x_b_minus_y_b = cp_model.NewIntVar(domain_z_b_minus_x_b_minus_y_b).WithName("z_b_minus_x_b_minus_y_b");
  LinearExpr z_b_minus_x_b_minus_y_b_expr;
  z_b_minus_x_b_minus_y_b_expr.AddTerm(x_b, -1);
  z_b_minus_x_b_minus_y_b_expr.AddTerm(y_b, -1);
  z_b_minus_x_b_minus_y_b_expr.AddConstant(z_b);
  cp_model.AddEquality(z_b_minus_x_b_minus_y_b, z_b_minus_x_b_minus_y_b_expr);

  const Domain domain_prod1(0, s_a * z_b);
  const IntVar prod1 = cp_model.NewIntVar(domain_prod1).WithName("prod1");
  std::vector<IntVar> prod1_vars;
  prod1_vars.push_back(s_a_minus_x_a);
  prod1_vars.push_back(z_b_minus_x_b_minus_y_b);
  cp_model.AddProductEquality(prod1, prod1_vars);

  const Domain domain_prod2(0, s_b * z_a);
  const IntVar prod2 = cp_model.NewIntVar(domain_prod2).WithName("prod2");
  std::vector<IntVar> prod2_vars;
  prod2_vars.push_back(s_b_minus_x_b);
  prod2_vars.push_back(z_a_minus_x_a_minus_y_a);
  cp_model.AddProductEquality(prod2, prod2_vars);

  const Domain domain_diff(std::min(- s_a * z_b, - s_b * z_a),std::max(s_a * z_b, s_b * z_a));
  const IntVar diff = cp_model.NewIntVar(domain_diff).WithName("diff");
  LinearExpr difference;
  difference.AddTerm(prod1, 1);
  difference.AddTerm(prod2, -1);
  cp_model.AddEquality(diff, difference);

  const Domain domain_diff_abs(0, std::max(s_a * z_b, s_b * z_a));
  const IntVar diff_abs = cp_model.NewIntVar(domain_diff_abs).WithName("diff_abs");
  cp_model.AddAbsEquality(diff_abs, diff);

  const Domain domain_diff_abs_scaled(0, std::max(s_a * z_b, s_b * z_a)*oneOverEpsilon);
  const IntVar diff_abs_scaled = cp_model.NewIntVar(domain_diff_abs_scaled).WithName("diff_abs_scaled");
  LinearExpr diff_abs_scaled_expr;
  diff_abs_scaled_expr.AddTerm(diff_abs, oneOverEpsilon);
  cp_model.AddEquality(diff_abs_scaled, diff_abs_scaled_expr);

  const Domain domain_other_term(0, z_a * z_b);
  const IntVar other_term = cp_model.NewIntVar(domain_other_term).WithName("other_term");
  std::vector<IntVar> other_term_vars;
  other_term_vars.push_back(z_a_minus_x_a_minus_y_a);
  other_term_vars.push_back(z_b_minus_x_b_minus_y_b);
  cp_model.AddProductEquality(other_term, other_term_vars);

  cp_model.AddGreaterThan(diff_abs_scaled, other_term);

  const Domain domain_objective(obj_lb, obj_ub);
  const IntVar objective = cp_model.NewIntVar(domain_objective).WithName("objective");
  LinearExpr objective_expr;
  objective_expr.AddTerm(x_a, 1);
  objective_expr.AddTerm(y_a, 1);
  objective_expr.AddTerm(x_b, 1);
  objective_expr.AddTerm(y_b, 1);
  cp_model.AddEquality(objective, objective_expr);

  cp_model.Minimize(objective);
  
  //cp_model.AddNotEqual(x, y);

  Model model;
  const CpSolverResponse response = SolveCpModel(cp_model.Build(), &model);
  
  if (response.status() == CpSolverStatus::OPTIMAL) {
    LOG(INFO) << "  x_a = " << SolutionIntegerValue(response, x_a);
    LOG(INFO) << "  x_b = " << SolutionIntegerValue(response, x_b);
    LOG(INFO) << "  y_a = " << SolutionIntegerValue(response, y_a);
    LOG(INFO) << "  y_b = " << SolutionIntegerValue(response, y_b);

    LOG(INFO) << "  prod1 = " << SolutionIntegerValue(response, prod1);
    LOG(INFO) << "  prod2 = " << SolutionIntegerValue(response, prod2);
    LOG(INFO) << "  diff = " << SolutionIntegerValue(response, diff);
    LOG(INFO) << "  diff_abs = " << SolutionIntegerValue(response, diff_abs);
    LOG(INFO) << "  diff_abs_scaled = " << SolutionIntegerValue(response, diff_abs_scaled);
    LOG(INFO) << "  other_term = " << SolutionIntegerValue(response, other_term);
    
    LOG(INFO) << "  objective = " << SolutionIntegerValue(response, objective);

    LOG(INFO) <<  " unfairness over found subset is " << (float) fabs((float) ((float)(s_a - SolutionIntegerValue(response, x_a) )/(float)(z_a - SolutionIntegerValue(response, x_a) - SolutionIntegerValue(response, y_a))) - (float) ((float)(s_b - SolutionIntegerValue(response, x_b))/(float)(z_b - SolutionIntegerValue(response, x_b) - SolutionIntegerValue(response, y_b)))) ;
  }
 /* model.Add(NewFeasibleSolutionObserver([&](const CpSolverResponse& r) {
    LOG(INFO) << "Solution " << num_solutions;
    LOG(INFO) << "  x_a = " << SolutionIntegerValue(r, x_a);
    LOG(INFO) << "  x_b = " << SolutionIntegerValue(r, x_b);
    LOG(INFO) << "  y_a = " << SolutionIntegerValue(r, y_a);
    LOG(INFO) << "  y_b = " << SolutionIntegerValue(r, y_b);

    num_solutions++;
  })); 

  // Tell the solver to enumerate all solutions.
  SatParameters parameters;
  parameters.set_enumerate_all_solutions(true);
  model.Add(NewSatParameters(parameters));
  const CpSolverResponse response = SolveCpModel(cp_model.Build(), &model);

  LOG(INFO) << "Number of solutions found: " << num_solutions; */
}

}  // namespace sat
}  // namespace operations_research

int main() {
    int s_a = 500;
    int s_b = 500;
    int z_a = 1000;
    int z_b = 1000;
    float epsilon = 0.05;

    int obj_lb = -1;
    int obj_ub = -1;

    std::cout <<  " c Instance data :  " <<  	std::endl;

    std::cout <<  " c s_a "  << s_a <<	std::endl;
    std::cout <<  " c z_a "  << z_a <<	std::endl;
    std::cout <<  " c s_b "  << s_b <<	std::endl;
    std::cout <<  " c z_b "  << z_b <<	std::endl;

    std::cout <<  " c Unfairness tolerance (should be in [0,1]): "  << epsilon << 	std::endl;
    std::cout <<  " c Unfairness over dataset (should be in [0,1]): "  << (float) fabs((float) ((float)s_a/(float)z_a) - (float) ((float)s_b/(float)z_b)) << 	std::endl;
  operations_research::sat::AuditSampleFairness(s_a, z_a, s_b, z_b, epsilon, obj_lb, obj_ub);

  return EXIT_SUCCESS;
}