import os
import pandas as pd
import re

def main():
    
    # ensemble the predicted labels

    models = ['res_net', 'eff_net', 'vgg16']
    pred_probs = None

    for model in models:

        df = pd.read_csv('results/predictions/predictions-' + model + '.csv')

        if pred_probs is None:
            pred_probs = df['preds']
        else:
            pred_probs += df['preds']
        
    pred_probs /= len(models)

    pred_labels = [1 if prob > 0.5 else 0 for prob in pred_probs]

    # retrieve image names from file names
    lst_files_test = os.listdir('data/processed/test/benign')
    num_benign_test = len(lst_files_test)
    lst_files_malignant_test = os.listdir('data/processed/test/malignant')
    num_malignant_test = len(lst_files_malignant_test)
    lst_files_test.extend(lst_files_malignant_test)

    filenames_clean = [re.sub('_downsampled', '', fname[:-4]) for fname in lst_files_test]

    filenames_clean

    # save submission

    target = pred_labels
    df_labels = pd.DataFrame({'image_name': filenames_clean, 'target': target})
    df_probs = pd.DataFrame({'image_name': filenames_clean, 'probs': pred_probs})

    df_labels.to_csv('results/predictions/ensembled_labels.csv', index = False)
    df_probs.to_csv('results/predictions/ensembled_probs.csv', index = False)
    
    print(f"Predicted ensembled class labels saved to 'results/predictions/ensembled_labels.csv'.")
    print(f"Predicted ensembled class probabilities saved to 'results/predictions/ensembled_probs.csv'.")
    

if __name__ == '__main__':
    
    main()
    