# Naive Bayes para valores numéricos em 100 linhas by elf
# Gaussian Naive Bayes from J. Brownlee: Code Machine Learning Algorithms From Scratch
import csv, random, math
import numpy #para pegar desvio padrão 
inf=1e-20

def blank(vet):
    if vet==[]: return True
    for x in vet:
        if "#" in x: return True
    return False
def x2float(x): # senão retorna ele mesmo 
    try: return float(x)
    except ValueError:return x
    
def loadCsv(filename):
    lines = csv.reader(open("pima-indians-diabetes.csv.txt", "r"))
    DATA = list(lines) [1:]
    while blank(DATA[0]): DATA.pop(0) 
    #print(DATA[:15]);exit(1)
    for i in range(len(DATA)):
        DATA[i] = [x2float(x) for x in DATA[i]]
    return DATA

random.seed(37) 
def splitData(DATA, percent):
    random.shuffle(DATA); meio=int(len(DATA) * percent)
    return (DATA[:meio],DATA[meio:])

Data1=[(1,2,'a'),(1,1,'a'),(2,1,'b'),(1,1,'b'), (1,1,'c'),(3,2,'c')]
def groupByClass(DATA):
    # assume classe ultimo atributo: -1
    colClass=[lin[-1] for lin in DATA]
    groups={}
    for c in set(colClass):
       groups[c]= [lin for lin in DATA if lin[-1]==c]
    return groups
#print(groupByClass(Data1)); exit(1)
'''
{'b': [(2, 1, 'b'),
       (1, 1, 'b')],
 'a': [(1, 2, 'a'),
       (1, 1, 'a')],
 'c': [(1, 1, 'c'),
       +(3, 2, 'c')]}
'''

vet=[1,2,3,5,6,5,5,7,3,3,1]
def med(vet):return sum(vet)/len(vet)
def stdev(vet):
    m=med(vet);soma=0
    for x in vet:soma +=(x-m)**2
    return math.sqrt(soma/len(vet))
#print('stdev:', numpy.std(vet), stdev(vet));exit(1)

#medDevCol  = totalizações por coluna
#modeloProb = {medDevCol para cada classe}     

# zip Agrupa por coluna 
#print(list(zip(* Data1)));exit(1)
# [(1, 1, 2, 3),  (2, 1, 1, 2), ('a', 'a', 'b', 'c')]

def medDevCol(DATA):
   return [(med(x), stdev(x)) for x in list(zip(*DATA))[:-1]]

# modByClass =  {medDevByClass = por coluna por classe} 
def modByClass(DATA):
   group = groupByClass(DATA)
   modeloC = {}
   for classe, insts in group.items(): modeloC[classe] = medDevCol(insts)
   return modeloC
#print(Data1, '\n medDev Col:\n', medDevCol(Data1),'\n modelo by class:\n',modByClass(Data1)); exit(1)

#prob=probabilidade 
def prob(x, med, stdev):
    expo = math.exp(-((x-med)**2)/(inf+2*stdev**2))
    return (1 / (inf+math.sqrt(2*math.pi) * stdev)) * expo
# print(prob(5, 7, 1)); exit(1)

# aplica o modelo numa instancia qq
def runModelo(modeloProb, vet):
    probs = {}
    for classe, medDev in modeloProb.items():
        probs[classe] = 1 # neutro na mult
        for i in range(len(medDev)):
            med, dev = medDev[i]; x = vet[i]
            probs[classe] *= prob(x, med, dev)
    return probs
#print(Data1, '\n para [1,2]:\n', runModelo(modByClass(Data1),[1,2])); 
#print(Data1, '\n para [3,1]:\n', runModelo(modByClass(Data1),[3,1])); exit(1)
			
def predict(modProb, vet):
   probs = runModelo(modProb, vet)
   #print(probs.items()); exit(1)
   maxClass=max(probs.items(), key=lambda x:x[1])
   return maxClass[0]
#print(max([(1,2.2), (3,3.4), (0,6.1), (7,1.0)], key=lambda x:x[1]));exit(1)  
def predictSet(modProb, testSet):
   return [predict(modProb, i) for i in testSet] 
 
def getAccuracy(testSet, predictSet):
    nro_ok = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictSet[i]:nro_ok += 1
    print(nro_ok,float(len(testSet)))
    return (nro_ok/float(len(testSet))) * 100.0

def main():
    #filename = 'pima-indians-diabetes.csv'
    filename = 'iris.csv'
    percent = 0.67; DATA = loadCsv(filename)
    trainingSet, testSet = splitData(DATA, percent)
    print('DATA = %d linhas,  treino=%d e test=%d' %
       (len(DATA), len(trainingSet), len(testSet)))
    print('data ex:\n',DATA[:3])
    # prepare model
    modeloProb = modByClass(trainingSet)
    # test model
    predict_y  = predictSet(modeloProb, testSet)
    accuracy   = getAccuracy(testSet, predict_y)
    print('Accuracy: %5.2f' % accuracy)
main()
