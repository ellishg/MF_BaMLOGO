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
{function: {algorithm: {'Costs': costs,
                        'Values': values,
                        'QueryPoints': queryPoints,
                        'TrueOptima': trueOptima,
                        'BestQuery': alg.bestQuery}}}
'''
def plot(results):
    import matplotlib.pyplot as plt
    import numpy as np
    for i, (fn, fnResults) in enumerate(results.items()):
        plt.figure(i)
        for alg, algResults in fnResults.items():
            costs = algResults['Costs']
            trueOptima = algResults['TrueOptima']
            errors = trueOptima - np.array(algResults['Values'])
            plt.plot(costs, errors, label=alg)
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
