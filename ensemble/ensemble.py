from sklearn.metrics import classification_report, confusion_matrix
from Abestxlnet import *
from Bbestxlnet import *
from Cbestxlnet import *
from Abestbert import *
from Bbestbert import *
from Cbestbert import *
from Abestroberta import *
from Bbestroberta import *
from Cbestroberta import *
def voting_ensemble(list1, list2, list3, eval_list):
    ensemblepredlist = []
    for i in range(len(list1)):
        pred1 = list1[i]
        pred2 = list2[i]
        pred3 = list3[i]
        predictionslist = [pred1, pred2, pred3]
        finalpred = max(set(predictionslist), key = predictionslist.count)
        ensemblepredlist.append(finalpred)
    #print(ensemblepredlist)K
    #print(eval_list)
    print("VOTING ENSEMBLE:")
    print(classification_report(eval_list, ensemblepredlist))
    print(confusion_matrix(eval_list, ensemblepredlist))


def weighted_ensemble(model1_outputs, model2_outputs, model3_outputs, eval_list):
    ensemblefinallist = []
    for i in range(len(model1_outputs)):
        ensemblepredlist = []
        pred1list = model1_outputs[i]
        pred2list = model2_outputs[i]
        pred3list = model3_outputs[i]
        for x in range(len(pred1list)):
            ensemblepredlist.append(pred1list[x] + pred2list[x] + pred3list[x])
        ensemblefinallist.append(ensemblepredlist)
    #print(ensemblefinallist)
    finalpredictionslist = []
    for a in ensemblefinallist:
        finalpredictionslist.append(a.index(max(a)))
    #print("Predicted classes of ensemble:", finalpredictionslist)
    #print("Classes of evaluation:", eval_list)
    print("WEIGHTED ENSEMBLE:")
    print(classification_report(eval_list, finalpredictionslist))
    print(confusion_matrix(eval_list, finalpredictionslist))



def main():
    print("TASK A:")
    weights1, preds1, eval_list = Abestroberta()
    weights2, preds2, eval_list = Abestxlnet()
    weights3, preds3, eval_list = Abestbert()
    voting_ensemble(preds1, preds2, preds3, eval_list)
    weighted_ensemble(weights1, weights2, weights3, eval_list)
    #print("TASK B:")
    #weights1, preds1, eval_list = Bbestroberta()
    #weights2, preds2, eval_list = Bbestxlnet()
    #weights3, preds3, eval_list = Bbestbert()
    #voting_ensemble(preds1, preds2, preds3, eval_list)
    #weighted_ensemble(weights1, weights2, weights3, eval_list)
    #print("TASK C:")
    #weights1, preds1, eval_list = Cbestroberta()
    #weights2, preds2, eval_list = Cbestxlnet()
    #weights3, preds3, eval_list = Cbestbert()
    #voting_ensemble(preds1, preds2, preds3, eval_list)
    #weighted_ensemble(weights1, weights2, weights3, eval_list)


if __name__ == '__main__':
    main()
