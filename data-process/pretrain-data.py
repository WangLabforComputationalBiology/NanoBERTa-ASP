import pandas as pd
import os


def filetoseq():
    file_name_list = os.listdir()
    d_file = []
    for file in file_name_list:
        if '.csv' not in file:
            d_file.append(file)
    for df in d_file:
        file_name_list.remove(df)
    del d_file
    print(file_name_list)
    for element in file_name_list:
        csv_to_xlsx_pd(element)


def csv_to_xlsx_pd(f_path):
    csv = pd.read_csv(f_path, header=1)
    for index, row in csv.iterrows():
        len_cdr3 = len(row['cdr3_aa'])
        seq = row['sequence_alignment_aa']
        if 'X' not in row['sequence_alignment_aa']:
            result1 = is_in(row['sequence_alignment_aa'], str(row['cdr1_aa']))
            result3 = is_in(row['sequence_alignment_aa'], str(row['cdr3_aa']))
            total = len(row['sequence_alignment_aa'])
            if result1 >= 20 and (total - len_cdr3 - result3) >= 10:
                with open('mlm_seq.txt', mode='a+') as f:
                    f.write(seq + '\n')
                del seq
            else:
                pass
        else:
            pass


def is_in(full_str, sub_str):
    try:
        full_str.index(sub_str)
        return full_str.index(sub_str)
    except ValueError:
        return False


if __name__ == '__main__':
    filetoseq()
