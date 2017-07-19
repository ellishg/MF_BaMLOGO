import sys

def main(argv):
    optionExample = (argv[0] + ' -h'
                               ' -f <test_function>'
                               ' -a <algorithm>'
                               ' -r <budget>'
                               ' -n <num_runs>'
                               ' -o <output>'
                               ' -v <verbose_level>')
    testFunction = 'Hartmann-3D'
    algorithm = 'MF-BaMLOGO'
    budget = 50
    outputDir = None
    numRuns = 1
    import getopt
    try:
        opts, args = getopt.getopt(argv[1:], 'hf:a:r:n:o:v:')
    except getopt.GetoptError:
        print(optionExample)
        exit(1)
    for opt, arg in opts:
        if opt == '-h':
            print(optionExample)
            exit(0)
        elif opt == '-v':
            import logging
            if arg == '0':
                logging.basicConfig(level=logging.ERROR)
            elif arg == '1':
                logging.basicConfig(level=logging.INFO)
            else:
                logging.basicConfig(level=logging.DEBUG)
        elif opt == '-f':
            testFunction = arg
        elif opt == '-a':
            algorithm = arg
        elif opt == '-r':
            budget = float(arg)
        elif opt == '-n':
            numRuns = int(arg)
        elif opt == '-o':
            outputDir = arg

    if testFunction == 'Hartmann-3D':
        from mf_test_functions import getMFHartmann
        numFidelities = 3
        fn = getMFHartmann(numFidelities, 3)
        costs = [10. ** -(numFidelities - f - 1) for f in range(numFidelities)]
        lows = 3 * [0.]
        highs = 3 * [1.]
        trueOptima = 3.86278
    elif testFunction == 'Hartmann-6D':
        from mf_test_functions import getMFHartmann
        numFidelities = 4
        fn = getMFHartmann(numFidelities, 6)
        costs = [10. ** -(numFidelities - f - 1) for f in range(numFidelities)]
        lows = 6 * [0.]
        highs = 6 * [1.]
        trueOptima = 3.32237
    elif testFunction == 'Park1-4D':
        from mf_test_functions import park1, lowFidelityPark1
        def fn(x, f):
            if f == 0:
                return lowFidelityPark1(x)
            elif f == 1:
                return park1(x)
            else:
                return float('nan')
        costs = [0.1, 1.]
        lows = 4 * [0.]
        highs = 4 * [1.]
        trueOptima = 25.5893
    elif testFunction == 'Park2-4D':
        from mf_test_functions import park2, lowFidelityPark2
        def fn(x, f):
            if f == 0:
                return lowFidelityPark2(x)
            elif f == 1:
                return park2(x)
            else:
                return float('nan')
        costs = [0.1, 1.]
        lows = 4 * [0.]
        highs = 4 * [1.]
        trueOptima = 5.92604
    elif testFunction == 'CurrinExponential-2D':
        from mf_test_functions import (currinExponential,
                                       lowFideliltyCurrinExponential)
        def fn(x, f):
            if f == 0:
                return lowFideliltyCurrinExponential(x)
            elif f == 1:
                return currinExponential(x)
            else:
                return float('nan')
        costs = [0.1, 1.]
        lows = [0., 0.]
        highs = [1., 1.]
        trueOptima = 13.7987
    elif testFunction == 'BadCurrinExponential-2D':
        from mf_test_functions import currinExponential
        def fn(x, f):
            if f == 0:
                return -currinExponential(x)
            elif f == 1:
                return currinExponential(x)
            else:
                return float('nan')
        costs = [0.1, 1.]
        lows = [0., 0.]
        highs = [1., 1.]
        trueOptima = 13.7987
    elif testFunction == 'Borehole-8D':
        from mf_test_functions import borehole, lowFidelityBorehole
        def fn(x, f):
            if f == 0:
                return lowFidelityBorehole(x)
            elif f == 1:
                return borehole(x)
            else:
                return float('nan')
        costs = [0.1, 1.]
        lows = [.05, 100., 63070., 990., 63.1, 700., 1120., 9855.]
        highs = [.15, 50000., 115600., 1110., 116., 820., 1680., 12045.]
        trueOptima = 309.523221
    elif testFunction == 'SCALE-8D':
        from eval_scale import evalWeightsSCALE
        def fn(x, f):
            if f == 0:
                return evalWeightsSCALE(x, 30)
            elif f == 1:
                return evalWeightsSCALE(x, 150)
            elif f == 2:
                return evalWeightsSCALE(x, 569)
            else:
                return float('nan')
        costs = [0.1, 0.25, 1.0]
        lows = 38 * [0.]
        highs = 38 * [1.]
        trueOptima = 1.
    else:
        print('Unknown test function.')
        exit(1)

    results = runAlgorithm(testFunction, fn, costs, lows, highs, trueOptima,
                                budget, algorithm, numRuns)

    if outputDir:
        with open(outputDir, 'w') as outFile:
            import json
            json.dump(results, outFile)
    else:
        best = results[testFunction][algorithm]['BestQuery']
        print('Found f{0} = {1}'.format(*best))
        from plot import plot
        plot(results)

def runAlgorithm(functionName, fn,
                 costs, lows, highs,
                 trueOptima, budget, algorithm, numRuns):
    from mf_bamlogo import MF_BaMLOGO, ObjectiveFunction
    objectiveFunction = ObjectiveFunction(fn, costs, lows, highs)

    runs = []
    for _ in range(numRuns):
        alg = MF_BaMLOGO(objectiveFunction, initNumber=10, algorithm=algorithm)
        costs, values, queryPoints = alg.maximize(budget=budget,
                                                  ret_data=True)
        runs.append({'Costs': costs,
                     'Values': values,
                     'QueryPoints': queryPoints,
                     'BestQuery': alg.bestQuery()
                     })

    results = {functionName: {algorithm: {'TrueOptima': trueOptima,
                                          'Runs': runs}}}
    return results

if __name__ == '__main__':
    main(sys.argv)
