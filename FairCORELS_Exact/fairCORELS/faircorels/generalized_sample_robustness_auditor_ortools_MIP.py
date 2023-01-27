def audit_robustness_or_tools_MIP(s_a, s_b, z_a, z_b, epsilon, debug, objective_lower_bound, objective_upper_bound):
    from ortools.linear_solver import pywraplp
    solver = pywraplp.Solver('Filtering Solver', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    oneOverEpsilon = int(1/epsilon)
    #print(oneOverEpsilon)

    x_a = solver.IntVar(0, s_a, 'x_a')
    x_b = solver.IntVar(0, s_b, 'x_b')
    y_a = solver.IntVar(0, z_a - s_a, 'y_a')
    y_b = solver.IntVar(0, z_b - s_b, 'y_b')

    big_number = z_a+z_b
    x_a_prime = solver.IntVar(0, big_number, 'x_a_prime')
    x_b_prime = solver.IntVar(0, big_number, 'x_b_prime')
    y_a_prime = solver.IntVar(0, big_number, 'y_a_prime')
    y_b_prime = solver.IntVar(0, big_number, 'y_b_prime')
    
    a_numerator = solver.IntVar(0, s_a + big_number, 'a_numerator')
    b_numerator = solver.IntVar(0, s_b + big_number, 'b_numerator')
    a_denominator = solver.IntVar(1, z_a + 2*big_number, 'a_denominator')
    b_denominator = solver.IntVar(1, z_b + 2*big_number, 'b_denominator')
    solver.Add(a_numerator == s_a - x_a + x_a_prime)
    solver.Add(b_numerator == s_b - x_b + x_b_prime)
    solver.Add(a_denominator == z_a - x_a - y_a + x_a_prime + y_a_prime)
    solver.Add(b_denominator == z_b - x_b - y_b + x_b_prime + y_b_prime)

    #prod1 = solver.IntVar(0, (s_a + big_number) * (z_b + 2*big_number), 'prod1')
    #prod2 = solver.IntVar(0, (s_b + big_number) * (z_a + 2*big_number), 'prod2')
   # solver.AddMultiplicationEquality(prod1, [a_numerator, b_denominator])
    #solver.Add(prod1 == a_numerator * b_denominator)
    #solver.AddMultiplicationEquality(prod2, [b_numerator, a_denominator])

    ratio1 = solver.NumVar(0, 1, "ratio1")
    solver.Add(ratio1 == a_numerator * a_denominator)
    diff = solver.IntVar(min(- (s_a + big_number) * (z_b + 2*big_number), (s_b + big_number) * (z_a + 2*big_number)), max((s_a + big_number) * (z_b + 2*big_number), (s_b + big_number) * (z_a + 2*big_number)), 'diff')
    diff_abs = solver.IntVar(0,  max((s_a + big_number) * (z_b + 2*big_number), (s_b + big_number) * (z_a + 2*big_number)), 'diff_abs')
    diff_abs_scaled = solver.IntVar(0,  max((s_a + big_number) * (z_b + 2*big_number), (s_b + big_number) * (z_a + 2*big_number))*oneOverEpsilon, 'diff_abs_scaled')
    solver.Add(diff == prod1 - prod2)
    solver.AddAbsEquality(diff_abs, diff)
    solver.Add(diff_abs_scaled == oneOverEpsilon * diff_abs)

    other_term = solver.IntVar(0, ( z_a + 2*big_number) * (z_b + 2*big_number), "other_term")
    model.AddMultiplicationEquality(other_term, [a_denominator, b_denominator])
    #bool_for_or = model.NewIntVar(0, 1, 'disjunction_switch')
    #{model.Add(diff_abs)
    solver.Add(other_term < diff_abs_scaled)
    
    # included in z_i_minus_x_i_minus_y_i domains
    #model.Add(x_a + y_a < z_a)
    #model.Add(x_b + y_b < z_b)

    # objective bounds
    obj_lb = 0
    obj_ub = z_a + z_b - 2
    if objective_lower_bound != -1:
        obj_lb = objective_lower_bound
    if objective_upper_bound != -1:
        obj_ub = objective_upper_bound

    # objective
    objective = solver.IntVar(obj_lb, obj_ub, 'objective')
    solver.Add(objective == x_a+x_b+y_a+y_b +x_a_prime + y_a_prime + x_b_prime + y_b_prime )
    solver.Minimize(objective)

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        if debug:
            print('Minimum of objective function: %i' % solver.ObjectiveValue())
            print()
            print('x_a value: ', solver.Value(x_a))
            print('y_a value: ', solver.Value(y_a))
            print('x_b value: ', solver.Value(x_b))
            print('y_b value: ', solver.Value(y_b))
            print('x_a_prime value: ', solver.Value(x_a_prime))
            print('y_a_prime value: ', solver.Value(y_a_prime))
            print('x_b_prime value: ', solver.Value(x_b_prime))
            print('y_b_prime value: ', solver.Value(y_b_prime))
        return solver.Value(x_a), solver.Value(y_a), solver.Value(x_b), solver.Value(y_b), solver.Value(x_a_prime), solver.Value(y_a_prime), solver.Value(x_b_prime), solver.Value(y_b_prime), solver.ObjectiveValue(), 0
    elif status == pywraplp.Solver.INFEASIBLE:
        return -1, -1, -1, -1, -1, -1, -1, -1, -1, 1
    elif status == pywraplp.Solver.FEASIBLE:
        #if debug:
        print('Found (non optimal) objective function: %i' % solver.ObjectiveValue())
        print()
        print('x_a value: ', solver.Value(x_a))
        print('y_a value: ', solver.Value(y_a))
        print('x_b value: ', solver.Value(x_b))
        print('y_b value: ', solver.Value(y_b))
        return solver.Value(x_a), solver.Value(y_a), solver.Value(x_b), solver.Value(y_b), solver.Value(x_a_prime), solver.Value(y_a_prime), solver.Value(x_b_prime), solver.Value(y_b_prime), solver.ObjectiveValue(), 2
    else:
        print("unknown status, exiting.")
        exit()
