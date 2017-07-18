import sys
from pathlib import Path
import json

def main(dataDir='data/'):
    results = dict()
    dataPath = Path(dataDir)
    for i, dataFile in enumerate(dataPath.iterdir()):
        if dataFile.is_file():
            print(dataFile)
            with dataFile.open() as f:
                jsonDict = json.loads(f.read())
                for fn, fnResults in jsonDict.items():
                    if fn not in results:
                        results[fn] = fnResults
                    else:
                        results[fn].update(fnResults)
    plot(results)

'''
runs.append({'Costs': costs,
             'Values': values,
             'QueryPoints': queryPoints,
             'BestQuery': alg.bestQuery()
             })

{functionName: {algorithm: {'TrueOptima': trueOptima,
                            'Runs': runs}}}
'''
def plot(results):
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    for i, (fn, fnResults) in enumerate(results.items()):
        plt.figure(i)
        for alg, algResults in fnResults.items():
            trueOptima = algResults['TrueOptima']
            errorBins = dict()  # Find a list of the means of each run
            for run in algResults['Runs']:
                runErrorBins = dict()
                costs = run['Costs']
                errors = trueOptima - np.array(run['Values'])
                for c, e in zip(costs, errors):
                    _bin = math.floor(c / 10) * 10
                    if _bin not in runErrorBins:
                        runErrorBins[_bin] = [e]
                    else:
                        runErrorBins[_bin].append(e)
                # Get the mean errors of the run
                for _bin, es in runErrorBins.items():
                    if _bin not in errorBins:
                        errorBins[_bin] = [np.mean(es)]
                    else:
                        errorBins[_bin].append(np.mean(es))

            costValues = list(errorBins.keys())
            errorValues = list(errorBins.values())
            means = np.array([np.mean(es) for es in errorValues])
            stds = np.array([np.std(es) for es in errorValues])
            plt.plot(costValues, means, label=alg)
            plt.fill_between(costValues,
                             means - stds,
                             means + stds,
                             color='#CCCCCC')
        plt.legend()
        plt.title(fn)
        plt.xlabel('Cumulative Cost')
        plt.ylabel('Simple Regret')
        plt.yscale('log')
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        main()
