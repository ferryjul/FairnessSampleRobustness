def audit_generalized_robustness_or_tools_cp(s_a, s_b, z_a, z_b, epsilon, debug, objective_lower_bound, objective_upper_bound):
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    oneOverEpsilon = int(1/epsilon)
    print(oneOverEpsilon)

    x_a = model.NewIntVar(0, s_a, 'x_a')
    x_b = model.NewIntVar(0, s_b, 'x_b')
    y_a = model.NewIntVar(0, z_a - s_a, 'y_a')
    y_b = model.NewIntVar(0, z_b - s_b, 'y_b')

    big_number = z_a+z_b
    x_a_prime = model.NewIntVar(0, big_number, 'x_a_prime')
    x_b_prime = model.NewIntVar(0, big_number, 'x_b_prime')
    y_a_prime = model.NewIntVar(0, big_number, 'y_a_prime')
    y_b_prime = model.NewIntVar(0, big_number, 'y_b_prime')

    a_numerator = model.NewIntVar(0, s_a + big_number, 'a_numerator')
    b_numerator = model.NewIntVar(0, s_b + big_number, 'b_numerator')
    a_denominator = model.NewIntVar(1, z_a + 2*big_number, 'a_denominator')
    b_denominator = model.NewIntVar(1, z_b + 2*big_number, 'b_denominator')
    model.Add(a_numerator == s_a - x_a + x_a_prime)
    model.Add(b_numerator == s_b - x_b + x_b_prime)
    model.Add(a_denominator == z_a - x_a - y_a + x_a_prime + y_a_prime)
    model.Add(b_denominator == z_b - x_b - y_b + x_b_prime + y_b_prime)

    prod1 = model.NewIntVar(0, (s_a + big_number) * (z_b + 2*big_number), 'prod1')
    prod2 = model.NewIntVar(0, (s_b + big_number) * (z_a + 2*big_number), 'prod2')
    model.AddMultiplicationEquality(prod1, [a_numerator, b_denominator])
    model.AddMultiplicationEquality(prod2, [b_numerator, a_denominator])

    diff = model.NewIntVar(min(- (s_a + big_number) * (z_b + 2*big_number), (s_b + big_number) * (z_a + 2*big_number)), max((s_a + big_number) * (z_b + 2*big_number), (s_b + big_number) * (z_a + 2*big_number)), 'diff')
    diff_abs = model.NewIntVar(0,  max((s_a + big_number) * (z_b + 2*big_number), (s_b + big_number) * (z_a + 2*big_number)), 'diff_abs')
    diff_abs_scaled = model.NewIntVar(0,  max((s_a + big_number) * (z_b + 2*big_number), (s_b + big_number) * (z_a + 2*big_number))*oneOverEpsilon, 'diff_abs_scaled')
    model.Add(diff == prod1 - prod2)
    model.AddAbsEquality(diff_abs, diff)
    model.Add(diff_abs_scaled == oneOverEpsilon * diff_abs)

    other_term = model.NewIntVar(0, ( z_a + 2*big_number) * (z_b + 2*big_number), "other_term")
    model.AddMultiplicationEquality(other_term, [a_denominator, b_denominator])
    #bool_for_or = model.NewIntVar(0, 1, 'disjunction_switch')
    #{model.Add(diff_abs)
    model.Add(other_term < diff_abs_scaled)
    
    # included in z_i_minus_x_i_minus_y_i domains
    #model.Add(x_a + y_a < z_a)
    #model.Add(x_b + y_b < z_b)

    factor=min(oneOverEpsilon*100, 1000000)
    # objective bounds
    obj_lb = 0
    #obj_ub = z_a + z_b - 2
    obj_ub = factor
    if objective_lower_bound != -1:
        obj_lb = objective_lower_bound
    if objective_upper_bound != -1:
        obj_ub = objective_upper_bound

    # objective
   
    d_inter_d_prime = model.NewIntVar(0, z_a + z_b, 'd_inter_d_prime')
    d_union_d_prime = model.NewIntVar(z_a+z_b, z_a + z_b + 4*big_number, 'd_union_d_prime')
    d_inter_d_prime_scaled = model.NewIntVar(0, (z_a + z_b) * factor, 'd_inter_d_prime_scaled')

    model.Add(d_inter_d_prime ==  z_a + z_b - x_a - x_b - y_a - y_b)
    model.Add(d_union_d_prime ==  z_a + z_b + x_a_prime + x_b_prime + y_a_prime + y_b_prime)
    model.Add(d_inter_d_prime_scaled == factor * d_inter_d_prime)

    objective = model.NewIntVar(obj_lb, obj_ub, 'objective')

    #model.Add(objective == x_a+x_b+y_a+y_b + x_a_prime + y_a_prime + x_b_prime + y_b_prime )
    #model.Minimize(objective)

    model.AddDivisionEquality(objective, d_inter_d_prime_scaled, d_union_d_prime)
    model.Maximize(objective)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        if debug:
            print('Maximum of objective function: %i' % solver.ObjectiveValue())
            print()
            print('x_a value: ', solver.Value(x_a))
            print('y_a value: ', solver.Value(y_a))
            print('x_b value: ', solver.Value(x_b))
            print('y_b value: ', solver.Value(y_b))
            print('x_a_prime value: ', solver.Value(x_a_prime))
            print('y_a_prime value: ', solver.Value(y_a_prime))
            print('x_b_prime value: ', solver.Value(x_b_prime))
            print('y_b_prime value: ', solver.Value(y_b_prime))
            print('Sample robustness value: ', solver.Value(y_b_prime))
        return solver.Value(x_a), solver.Value(y_a), solver.Value(x_b), solver.Value(y_b), solver.Value(x_a_prime), solver.Value(y_a_prime), solver.Value(x_b_prime), solver.Value(y_b_prime), solver.ObjectiveValue(), 0
    elif status == cp_model.INFEASIBLE:
        return -1, -1, -1, -1, -1, -1, -1, -1, -1, 1
    elif status == cp_model.FEASIBLE:
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
