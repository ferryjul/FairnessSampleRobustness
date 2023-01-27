def audit_robustness_or_tools(s_a, s_b, z_a, z_b, epsilon, debug, objective_lower_bound, objective_upper_bound):
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    oneOverEpsilon = int(1/epsilon)
    #print(oneOverEpsilon)

    x_a = model.NewIntVar(0, s_a, 'x_a')
    x_b = model.NewIntVar(0, s_b, 'x_b')
    y_a = model.NewIntVar(0, z_a - s_a, 'y_a')
    y_b = model.NewIntVar(0, z_b - s_b, 'y_b')

    s_a_minus_x_a = model.NewIntVar(0, s_a, 's_a_minus_x_a')
    s_b_minus_x_b = model.NewIntVar(0, s_b, 's_b_minus_x_b')
    z_a_minus_x_a_minus_y_a = model.NewIntVar(1, z_a, 'z_a_minus_x_a_minus_y_a')
    z_b_minus_x_b_minus_y_b = model.NewIntVar(1, z_b, 'z_b_minus_x_b_minus_y_b')
    model.Add(s_a_minus_x_a == s_a - x_a)
    model.Add(s_b_minus_x_b == s_b - x_b)
    model.Add(z_a_minus_x_a_minus_y_a == z_a - x_a - y_a)
    model.Add(z_b_minus_x_b_minus_y_b == z_b - x_b - y_b)

    prod1 = model.NewIntVar(0, s_a * z_b, 'prod1')
    prod2 = model.NewIntVar(0, s_b * z_a, 'prod1')
    model.AddMultiplicationEquality(prod1, [s_a_minus_x_a, z_b_minus_x_b_minus_y_b])
    model.AddMultiplicationEquality(prod2, [s_b_minus_x_b, z_a_minus_x_a_minus_y_a])

    diff = model.NewIntVar(min(- s_a * z_b, - s_b * z_a), max(s_a * z_b, s_b * z_a), 'diff')
    diff_abs = model.NewIntVar(0, max(s_a * z_b, s_b * z_a), 'diff_abs')
    diff_abs_scaled = model.NewIntVar(0, max(s_a * z_b, s_b * z_a)*oneOverEpsilon, 'diff_abs_scaled')
    model.Add(diff == prod1 - prod2)
    model.AddAbsEquality(diff_abs, diff)
    model.Add(diff_abs_scaled == oneOverEpsilon * diff_abs)

    other_term = model.NewIntVar(0, z_a * z_b, "other_term")
    model.AddMultiplicationEquality(other_term, [z_a_minus_x_a_minus_y_a, z_b_minus_x_b_minus_y_b])
    #bool_for_or = model.NewIntVar(0, 1, 'disjunction_switch')
    #{model.Add(diff_abs)
    model.Add(other_term < diff_abs_scaled)
    
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
    objective = model.NewIntVar(obj_lb, obj_ub, 'objective')
    model.Add(objective == x_a+x_b+y_a+y_b )
    model.Minimize(objective)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        if debug:
            print('Minimum of objective function: %i' % solver.ObjectiveValue())
            print()
            print('x_a value: ', solver.Value(x_a))
            print('y_a value: ', solver.Value(y_a))
            print('x_b value: ', solver.Value(x_b))
            print('y_b value: ', solver.Value(y_b))
        return solver.Value(x_a), solver.Value(y_a), solver.Value(x_b), solver.Value(y_b), solver.ObjectiveValue(), 0
    elif status == cp_model.INFEASIBLE:
        return -1, -1, -1, -1, -1, 1
    elif status == cp_model.FEASIBLE:
        #if debug:
        print('Found (non optimal) objective function: %i' % solver.ObjectiveValue())
        print()
        print('x_a value: ', solver.Value(x_a))
        print('y_a value: ', solver.Value(y_a))
        print('x_b value: ', solver.Value(x_b))
        print('y_b value: ', solver.Value(y_b))
        return solver.Value(x_a), solver.Value(y_a), solver.Value(x_b), solver.Value(y_b), solver.ObjectiveValue(), 2
    else:
        print("unknown status, exiting.")
        exit()
