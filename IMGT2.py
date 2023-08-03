#!wget https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all_nano/
import Bio.PDB
import re
import pyarrow as pa
import pyarrow.parquet as pq
import random
import os
import glob
from tqdm import tqdm


def random_cut(split_ratio=(0.6, 0.2, 0.2)):
    print(split_ratio)
    assert sum(split_ratio) == 1.0, "切分比例之和应为1.0"
    # assert os.path.exists(input_dir), "输入文件夹不存在"
    # # 创建输出目录
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    pdb_files_names = glob.glob(os.path.join(input_dir, "*.pdb"))
    pdb_files = [os.path.basename(f) for f in pdb_files_names]
    file_count = len(pdb_files)
    print(file_count)
    random.shuffle(pdb_files)

    # 切分文件，并将切分后的文件复制到输出目录中
    train_files = pdb_files
    write_parquest(train_files, 'sc.parquet', path=input_dir)
    print('完成')


def write_parquest(pdb_files, parquet_name, path=None):  # 写入parquet
    data_dict = {'sequence': [],
                 'paratope_labels': [],
                 'PDB': [],
                 'nano_chain/ antigen_chain':[]
                 }
    for file in tqdm(pdb_files, position=0):
        result = analyse_PDB(file, path)
        if result:
            data_dict['sequence'].extend(result[0])
            data_dict['paratope_labels'].extend(result[1])
            data_dict['PDB'].extend(result[2])
            data_dict['nano_chain/ antigen_chain'].extend(result[3])
            pass
    table = pa.Table.from_pydict(data_dict)
    schema = table.schema
    options = {'compression': 'snappy', 'use_dictionary': False}
    with pq.ParquetWriter(parquet_name, schema, **options) as writer:
        writer.write_table(table)


def IMGTcmp(file):
    pattern = r"REMARK\s+5\s+(SINGLE|PAIRED_HL)\s+HCHAIN=(?P<hchain>\w+)\s+(?:LCHAIN=(?P<lchain>\w+)\s+)?AGCHAIN=(?P<agchain>\w+)\s+AGTYPE=(?P<agtype>\w+)"
    results = []
    with open(file) as f:
        remarks = []
        for line in f:
            if line.startswith("REMARK"):
                remarks.append(line.strip())
    for remark in remarks:
        match = re.match(pattern, remark)
        if match:
            # 将每个项的名称和值存储到字典中
            data = {
                "H": match.group("hchain"),
                "L": match.group("lchain") or 'None',
                "Antigen": match.group("agchain"),
            }
            # 打印字典
            results.append(data)
        else:
            pass
    return results


def analyse_PDB(file, path=None):  # 计算PDB的结合位点
    # 检查文件名
    pdb_name = file[0:4]
    if path:
        file = path+'/'+file
    imgt = IMGTcmp(file)
    print(imgt)
    parser = Bio.PDB.PDBParser()
    try:
        structure = parser.get_structure(pdb_name, file)
    except Exception as e:
        print(f"The current parameters are: {pdb_name}")
        return
    sequences = []
    label_lists = []
    temp_combined = []
    clist=[]
    listreturn=[]
    chain_list = [c.get_id() for c in structure[0]]
    for t in imgt:
        nanobody_char = t['H']
        antigen_char = t['Antigen']
        # 获取纳米抗体和抗原的链
        if antigen_char == 'NONE' or nanobody_char == antigen_char:
            return sequences, label_lists, temp_combined, clist
            pass
        nanobody_char = nanobody_char.lower() if nanobody_char not in chain_list else nanobody_char
        antigen_char = antigen_char.lower() if antigen_char not in chain_list else antigen_char
        try:
            nanobody = structure[0][nanobody_char.strip()]
            antigen = structure[0][antigen_char.strip()]
        except Exception as e:
            print(f"An error occurred: {e}")
            print(t)
            print(f"The current parameters are: {pdb_name}-{antigen_char}-{nanobody_char}")
            exit()
        # 查找抗体和抗原之间的接触面积
        ns = Bio.PDB.NeighborSearch(list(antigen.get_atoms()))
        contact_residues = []
        paratope_seq = ''
        sequence = ''
        for residue in nanobody:
            if Bio.PDB.is_aa(residue.get_resname(), standard=True):  # 判断是否是氨基酸
                sequence += Bio.PDB.Polypeptide.protein_letters_3to1[residue.get_resname()]
                for atom in residue.get_atoms():
                    close_atoms = ns.search(atom.coord, 4.5, level='A')  # 定义接触面积的阈值为4.5埃
                    if len(close_atoms) > 0:
                        paratope_seq += Bio.PDB.Polypeptide.protein_letters_3to1[residue.get_resname()]
                        contact_residues.append(residue)
                        break
                else:
                    paratope_seq += '-'
        contact_positions = [residue.get_id()[1] for residue in contact_residues]
        # print("#" * 20, nanobody_char, 'against', antigen_char, "#" * 20)
        if len(contact_positions):
            # print('Y')
            if len(paratope_seq) <= 145:
                print(paratope_seq)
                print(contact_positions, len(contact_positions))
                label_list_temp = ["N" if c == "-" else "P" for c in paratope_seq]
                a = sequence+paratope_seq
                if a not in listreturn:
                    listreturn.append(a)

                    sequences.append(sequence)
                    label_lists.append(label_list_temp)
                    temp_combined.append(pdb_name)
                    clist.append(nanobody_char+antigen_char)
                pass
            else:
                print(paratope_seq[0:145])
                print(contact_positions, len(contact_positions))
                label_list_temp = ["N" if c == "-" else "P" for c in paratope_seq[0:145]]
                a = sequence+paratope_seq
                if a not in listreturn:
                    listreturn.append(a)
                    sequences.append(sequence[0:145])
                    label_lists.append(label_list_temp)
                    temp_combined.append(pdb_name)
                    clist.append(nanobody_char+antigen_char)
                pass
            sequence = ''
            label_list_temp = ''
        else:
            # print("NONE")
            pass
    return sequences, label_lists, temp_combined, clist


input_dir = 'all_structures/imgt/ASP'


if __name__ == '__main__':
    random_cut((0.8, 0.1, 0.1))
    # print(analyse_PDB('6woz.pdb', 'all_structures/imgt'))
    pass