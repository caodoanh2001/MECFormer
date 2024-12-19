import pandas as pd

f = open('tcga_esca/esca_meta.csv', 'w')
esca_df = pd.read_csv('/home/compu/doanhbc/WSIs-classification/wsi_dataset_annotation/tcga_esca/fold0.csv')

train_samples = esca_df['train']
train_labels = esca_df['train_label']

val_samples = esca_df['val'].dropna()
val_labels = esca_df['val_label'].dropna()

test_samples = esca_df['test'].dropna()
test_labels = esca_df['test_label'].dropna()

write_line = 'case_id,slide_id,label'
f.write(write_line+'\n')

i = 0
for sample, label in zip(train_samples, train_labels):
    f.write(','.join(['patient_' + str(i), sample, str(int(label))]) + '\n')
    i+=1

for sample, label in zip(val_samples, val_labels):
    f.write(','.join(['patient_' + str(i), sample, str(int(label))]) + '\n')
    i+=1

for sample, label in zip(test_samples, test_labels):
    f.write(','.join(['patient_' + str(i), sample, str(int(label))]) + '\n')
    i+=1