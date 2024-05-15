from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

def ecg_PrintScore(true, pred, savePath=None, average='macro'):
    # savePath=None -> console, else to Result.txt
    if savePath == None:
        saveFile = None
    else:
        saveFile = open(savePath + "Result.txt", 'w')
        
    # Main scores
    F1 = metrics.f1_score(true, pred, average=None)
    print("Main scores:")
    print('Acc\tF1S\tKappa\tAF\tNormal\tOther\tNoisy', file=saveFile)
    print('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f' %
          (metrics.accuracy_score(true, pred),
           metrics.f1_score(true, pred, average=average),
           metrics.cohen_kappa_score(true, pred),
           F1[0], F1[1], F1[2], F1[3]),
          file=saveFile)
    # Classification report
    print("\nClassification report:", file=saveFile)
    print(metrics.classification_report(true, pred,
                                        target_names=['AF','Normal','Other','Noisy'],
                                        digits=3), file=saveFile)
    # Confusion matrix
    print('Confusion matrix:', file=saveFile)
    print(metrics.confusion_matrix(true,pred), file=saveFile)
    # Overall scores
    print('\n    Accuracy\t',metrics.accuracy_score(true,pred), file=saveFile)
    print(' Cohen Kappa\t',metrics.cohen_kappa_score(true,pred), file=saveFile)
    print('    F1-Score\t',metrics.f1_score(true,pred,average=average), '\tAverage =',average, file=saveFile)
    print('   Precision\t',metrics.precision_score(true,pred,average=average), '\tAverage =',average, file=saveFile)
    print('      Recall\t',metrics.recall_score(true,pred,average=average), '\tAverage =',average, file=saveFile)
    if savePath != None:
        saveFile.close()

def sleepedf_PrintScore(true, pred, savePath=None, average='macro'):
    # savePath=None -> console, else to Result.txt
    if savePath == None:
        saveFile = None
    else:
        saveFile = open(savePath + "Result.txt", 'w')
        
    # Main scores
    F1 = metrics.f1_score(true, pred, average=None)
    print("Main scores:")
    print("Main scores:", file=saveFile)
    print('Acc\tF1S\tKappa\tF1_W\tF1_N1\tF1_N2\tF1_N3\tF1_R')
    print('Acc\tF1S\tKappa\tF1_W\tF1_N1\tF1_N2\tF1_N3\tF1_R', file=saveFile)
    print('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f' %
          (metrics.accuracy_score(true, pred),
           metrics.f1_score(true, pred, average=average),
           metrics.cohen_kappa_score(true, pred),
           F1[0], F1[1], F1[2], F1[3], F1[4]))
    print('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f' %
          (metrics.accuracy_score(true, pred),
           metrics.f1_score(true, pred, average=average),
           metrics.cohen_kappa_score(true, pred),
           F1[0], F1[1], F1[2], F1[3], F1[4]),
          file=saveFile)
    # Classification report
    print("\nClassification report:")
    print("\nClassification report:", file=saveFile)
    print(metrics.classification_report(true, pred,
                                        target_names=['Wake','N1','N2','N3','REM'],
                                        digits=4))
    print(metrics.classification_report(true, pred,
                                        target_names=['Wake','N1','N2','N3','REM'],
                                        digits=4), file=saveFile)
    # Confusion matrix
    print('Confusion matrix:')
    print('Confusion matrix:', file=saveFile)
    print(metrics.confusion_matrix(true,pred))
    print(metrics.confusion_matrix(true,pred), file=saveFile)
    # Overall scores
    # print('\n    Accuracy\t',metrics.accuracy_score(true,pred), file=saveFile)
    # print(' Cohen Kappa\t',metrics.cohen_kappa_score(true,pred), file=saveFile)
    # print('    F1-Score\t',metrics.f1_score(true,pred,average=average), '\tAverage =',average, file=saveFile)
    # print('   Precision\t',metrics.precision_score(true,pred,average=average), '\tAverage =',average, file=saveFile)
    # print('      Recall\t',metrics.recall_score(true,pred,average=average), '\tAverage =',average, file=saveFile)
    if savePath != None:
        saveFile.close()

def har_PrintScore(true, pred, savePath=None, average='macro'):
    # savePath=None -> console, else to Result.txt
    if savePath == None:
        saveFile = None
    else:
        saveFile = open(savePath + "Result.txt", 'w')
        
    # Main scores
    F1 = metrics.f1_score(true, pred, average=None)
    print("Main scores:")
    print("Main scores:", file=saveFile)
    print('Acc\tF1S\tKappa\tF1_Walking\tF1_Upstairs\tF1_Downstairs\tF1_Standing\tF1_Sitting\tF1_Lying')
    print('Acc\tF1S\tKappa\tF1_Walking\tF1_Upstairs\tF1_Downstairs\tF1_Standing\tF1_Sitting\tF1_Lying', file=saveFile)
    print('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f' %
          (metrics.accuracy_score(true, pred),
           metrics.f1_score(true, pred, average=average),
           metrics.cohen_kappa_score(true, pred),
           F1[0], F1[1], F1[2], F1[3], F1[4], F1[5]))
    print('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f' %
          (metrics.accuracy_score(true, pred),
           metrics.f1_score(true, pred, average=average),
           metrics.cohen_kappa_score(true, pred),
           F1[0], F1[1], F1[2], F1[3], F1[4], F1[5]),
          file=saveFile)
    # Classification report
    print("\nClassification report:")
    print("\nClassification report:", file=saveFile)
    print(metrics.classification_report(true, pred,
                                        target_names=['Walking','Upstairs','Downstairs','Standing','Sitting','Lying'],
                                        digits=5))
    print(metrics.classification_report(true, pred,
                                        target_names=['Walking','Upstairs','Downstairs','Standing','Sitting','Lying'],
                                        digits=5), file=saveFile)
    # Confusion matrix
    print('Confusion matrix:')
    print('Confusion matrix:', file=saveFile)
    print(metrics.confusion_matrix(true,pred))
    print(metrics.confusion_matrix(true,pred), file=saveFile)
    # Overall scores
    # print('\n    Accuracy\t',metrics.accuracy_score(true,pred), file=saveFile)
    # print(' Cohen Kappa\t',metrics.cohen_kappa_score(true,pred), file=saveFile)
    # print('    F1-Score\t',metrics.f1_score(true,pred,average=average), '\tAverage =',average, file=saveFile)
    # print('   Precision\t',metrics.precision_score(true,pred,average=average), '\tAverage =',average, file=saveFile)
    # print('      Recall\t',metrics.recall_score(true,pred,average=average), '\tAverage =',average, file=saveFile)
    if savePath != None:
        saveFile.close()

def tusz_PrintScore(true, pred, savePath=None, average='macro'):
    # savePath=None -> console, else to Result.txt
    if savePath == None:
        saveFile = None
    else:
        saveFile = open(savePath + "Result.txt", 'w')
        
    # Main scores
    F1 = metrics.f1_score(true, pred, average=None)
    print("Main scores:")
    print('Acc\tF1S\tKappa\tCF\tGN\tAB\tCT', file=saveFile)
    print('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f' %
          (metrics.accuracy_score(true, pred),
           metrics.f1_score(true, pred, average=average),
           metrics.cohen_kappa_score(true, pred),
           F1[0], F1[1], F1[2], F1[3]),
          file=saveFile)
    # Classification report
    print("\nClassification report:", file=saveFile)
    print(metrics.classification_report(true, pred,
                                        target_names=['CF','GN','AB','CT'],
                                        digits=4), file=saveFile)
    # Confusion matrix
    print('Confusion matrix:', file=saveFile)
    print(metrics.confusion_matrix(true,pred), file=saveFile)
    # Overall scores
    print('\n    Accuracy\t',metrics.accuracy_score(true,pred), file=saveFile)
    print(' Cohen Kappa\t',metrics.cohen_kappa_score(true,pred), file=saveFile)
    print('    F1-Score\t',metrics.f1_score(true,pred,average=average), '\tAverage =',average, file=saveFile)
    print('   Precision\t',metrics.precision_score(true,pred,average=average), '\tAverage =',average, file=saveFile)
    print('      Recall\t',metrics.recall_score(true,pred,average=average), '\tAverage =',average, file=saveFile)
    if savePath != None:
        saveFile.close()

def PrintScore(type_names, true, pred, logits, savePath=None, average='macro'):
    if savePath == None:
        saveFile = None
    else:
        saveFile = open(savePath + "Result.txt", 'w')
    
    y_true_bin = label_binarize(true, classes=np.unique(true))
    auc_pr = average_precision_score(y_true_bin, logits, average='weighted')
    auc_roc = roc_auc_score(y_true_bin, logits, average='weighted')
    
    num_of_types = len(type_names)
    # Main scores
    F1 = metrics.f1_score(true, pred, average=None)
    print("Main scores:")
    print("Main scores:", file=saveFile)

    sentence1 = 'Acc\tF1S\tKappa\t'
    sentence2 = '%.4f\t%.4f\t%.4f\t' %  (metrics.accuracy_score(true, pred),
           metrics.f1_score(true, pred, average=average),
           metrics.cohen_kappa_score(true, pred) )

    for idx, type_name in enumerate(type_names):
        sentence1 += 'F1_{}\t'.format(type_name)
        sentence2 += '%.4f\t'% (F1[idx])
    print(sentence1)
    print(sentence1, file=saveFile)
    print(sentence2)
    print(sentence2, file=saveFile)

    print('\nAUPRC\tAUROC')
    print('\nAUPRC\tAUROC', file=saveFile)
    print('%.4f\t%.4f\t'%(auc_pr,auc_roc))
    print('%.4f\t%.4f\t'%(auc_pr,auc_roc),file=saveFile)
    
    # Classification report
    print("\nClassification report:")
    print("\nClassification report:", file=saveFile)
    print(metrics.classification_report(true, pred, target_names= type_names, digits=num_of_types-1))
    print(metrics.classification_report(true, pred, target_names= type_names, digits=num_of_types-1), file=saveFile)
    # Confusion matrix
    print('Confusion matrix:')
    print('Confusion matrix:', file=saveFile)
    print(metrics.confusion_matrix(true,pred))
    print(metrics.confusion_matrix(true,pred), file=saveFile)
    # Overall scores
    # print('\n    Accuracy\t',metrics.accuracy_score(true,pred), file=saveFile)
    # print(' Cohen Kappa\t',metrics.cohen_kappa_score(true,pred), file=saveFile)
    # print('    F1-Score\t',metrics.f1_score(true,pred,average=average), '\tAverage =',average, file=saveFile)
    # print('   Precision\t',metrics.precision_score(true,pred,average=average), '\tAverage =',average, file=saveFile)
    # print('      Recall\t',metrics.recall_score(true,pred,average=average), '\tAverage =',average, file=saveFile)
    if savePath != None:
        saveFile.close()