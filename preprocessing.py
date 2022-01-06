from os import listdir
import numpy as np
import pandas as pd

# data from multiple data sources are downloaded from PubChem and placed in data directory in seperate folders
data_dir = 'data/'
folder_names = sorted(listdir(data_dir))
summs = [data_dir+f+'/summary.txt' for f in folder_names]
descs = [data_dir+f+'/description.txt' for f in folder_names]
print(len(summs))
data = []
for file_counter in range(len(summs)):
    # file_counter = 0
    summ = summs[file_counter]
    with open(summ) as file:
        lines = []
        for line in file:
            lines.append(line.rstrip('\n'))
    block_idx = []
    for i in range(len(lines)):
        if lines[i] == '':
            block_idx.append(i)
    for block_counter in range(len(block_idx)):
        # block_counter = 0
        if block_counter != len(block_idx) - 1:
            current_lines = lines[block_idx[block_counter]+1:block_idx[block_counter+1]]
        else:
            current_lines = lines[block_idx[block_counter]+1:]
        block_number = current_lines[0].split('.')[0]
        if int(block_number) != block_counter + 1:
            print('Error in parsing block '+str(block_number)+' in '+summ)
        name = current_lines[0].lstrip(block_number+'.').lstrip()
        source = np.NaN
        aid = np.NaN
        sub_num = np.NaN
        active_num = np.NaN
        target = np.NaN
        for line in current_lines:
            if line == current_lines[0]:
                continue
            # extract data source
            if line.startswith('Source'):
                source = line.split(':')[1].lstrip()
            # extract bioassay AID
            elif line.startswith('AID'):
                aid = line.split(':')[1].lstrip()
            # extract number of molecules and active molecules
            elif line.startswith('Substance BioActivity'):
                dummy = line.split(':')[1].lstrip()
                dummy_num = dummy.split()
                for num_counter in range(len(dummy_num)):
                    if 'Active' in dummy_num[num_counter]:
                        active_num = int(dummy_num[num_counter-1])
                    if 'Tested' in dummy_num[num_counter]:
                        sub_num = int(dummy_num[num_counter-1])
            # extract target
            elif line.startswith('Protein Targets') or line.startswith('Protein Target'):
                target = line.split(':')[1].lstrip()
            else:
                print('UNUSED DATA:', line)
        data.append([aid, name, source, block_number, target, sub_num, active_num])

# Parse descriptions
data_desc = []
for file_counter in range(len(descs)):
    desc = descs[file_counter]
    with open(desc) as file:
        lines = []
        for line in file:
            lines.append(line.rstrip('\n'))
    block_idx = []
    for i in range(len(lines)):
        if lines[i] == '':
            block_idx.append(i)
    for block_counter in range(len(block_idx)):
        if block_counter != len(block_idx) - 1:
            current_lines = lines[block_idx[block_counter] + 1:block_idx[block_counter + 1]]
        else:
            current_lines = lines[block_idx[block_counter] + 1:]
        if '.' not in current_lines[0]:
            continue
        block_number = current_lines[0].split('.')[0]
        if int(block_number) != block_counter + 1:
            print('Error in parsing block ' + str(block_number) + ' in ' + desc)
        name = current_lines[0].lstrip(block_number + '.').lstrip()
        source = np.NaN
        aid = np.NaN
        description = np.NaN
        for line in current_lines:
            # line=current_lines[1]
            if line == current_lines[0]:
                continue
            if line.startswith('Source:') and '_||_' not in line:
                source = line.split(':')[1].lstrip()
            elif line.startswith('AID:') and '_||_' not in line:
                aid = line.split(':')[1].lstrip()
            else:
                # Rules for parsing descriptions from different data sources
                dummy_lines = line.split('_||_')
                if folder_names[file_counter] == 'Broad_Ins':
                    description = line.replace('_||_', ' ')
                elif folder_names[file_counter] == 'Emory':
                    description = ''
                    useless = ['Assay Overview', 'NIH Molecular Libraries Screening Centers Network [MLSCN]',\
                    'Emory Chemical Biology Discovery Center in MLSCN','Assay provider','MLSCN Grant']
                    for dummy_line in dummy_lines:
                        if not any(x in dummy_line for x in useless):
                            description = description + ' ' + dummy_line
                        else:
                            print('UNUSED DATA:', dummy_line)
                elif folder_names[file_counter] == 'ICCB':
                    description = ''
                    useless = ['This screen was conducted by']
                    for dummy_line in dummy_lines:
                        if not any(x in dummy_line for x in useless):
                            description = description + ' ' + dummy_line
                        else:
                            print('UNUSED DATA:', dummy_line)
                elif folder_names[file_counter] == 'John_Hopkins':
                    description = ''
                    useless = ['Data Source', 'BioAssay Type', 'Source (MLPCN Center Name)', 'Screening Center PI',\
                               'Center Affiliation', 'Network:', 'Assay provider:','Assay Provider:','Grant Proposal Number',
                               'Grant Proposal PI', 'Assay Implementation', 'Name:', 'External Assay ID:']
                    reference_flag = 0
                    for dummy_line in dummy_lines:
                        if 'References' in dummy_line or 'Reference' in dummy_line:
                            reference_flag = 1
                        if not reference_flag and not any(x in dummy_line for x in useless):
                            description = description + ' ' + dummy_line
                        else:
                            print('UNUSED DATA:', dummy_line)
                        if 'Keywords' in dummy_line:
                            reference_flag = 0
                elif folder_names[file_counter] == 'Ncats':
                    description = ''
                    reference_flag = 0
                    useless = ['NIH Molecular Libraries Probe Centers Network [MLPCN]', 'MLPCN Grant',
                               'Assay Provider', 'Assay Submitter (PI)', 'NIH Chemical Genomics Center [NCGC]']
                    for dummy_line in dummy_lines:
                        if 'References' in dummy_line or 'Reference' in dummy_line:
                            reference_flag = 1
                        if not reference_flag and not any(x in dummy_line for x in useless):
                            description = description + ' ' + dummy_line
                        else:
                            print('UNUSED DATA:', dummy_line)
                        if 'Keywords' in dummy_line:
                            reference_flag = 0
                elif folder_names[file_counter] == 'NMMLSC':
                    description = ''
                    reference_flag = 0
                    useless = ['University of New Mexico Assay Overview:', 'Assay Support:',
                               'PI:', 'PI Affiliation:', 'Screening Center PI:', 'Screening Lead:',
                               'Assay Implementation:', 'UNM Cheminformatics:', 'Chemistry:',
                               'Vanderbilt Specialized Chemistry Center PI:', 'Vanderbilt Chemistry Lead:',
                               'Assay Background and Significance:', 'Project Title:', 'Screening Center Manager:',
                               'Screening Center/PI:', 'Lead Biologist:', 'Screening Operations Team:',
                               'Chemistry Lead:', 'Specialized Chemistry Center:', 'Assay Support:',
                               'University of New Mexico Center for Molecular Discovery  PI:', 'Center PI:',
                               'Target Team Leader for the Center:', 'KU SCC Project Manager:',
                               'KU SCC Chemists on this project:', 'Assay provider:','Assay Provider:', 'Co-PI:', 'KU Specialized Chemistry Center PI:']
                    for dummy_line in dummy_lines:
                        if not any(x in dummy_line for x in useless):
                            description = description + ' ' + dummy_line
                        else:
                            print('UNUSED DATA:', dummy_line)
                elif folder_names[file_counter] == 'Sanford_Burnam':
                    description = ''
                    reference_flag = 0
                    useless = ['Data Source:', 'Source Affiliation:', 'Network:',
                               'NIH Molecular Libraries Probe Production Centers Network (MLPCN)',
                               'Grant Number:', 'Assay Provider:', 'Grant Proposal Number:']
                    for dummy_line in dummy_lines:
                        if 'REFERENCES' in dummy_line or 'References' in dummy_line:
                            reference_flag = 1
                        if not reference_flag and not any(x in dummy_line for x in useless):
                            description = description + ' ' + dummy_line
                        else:
                            print('UNUSED DATA:', dummy_line)
                elif folder_names[file_counter] == 'Scripps':
                    description = ''
                    reference_flag = 0
                    useless = ['Source (MLPCN Center Name):','Source (MLSCN Center Name):', 'Center Affiliation:', 'Assay Provider:',
                               'Network:', 'Grant Proposal Number', 'Grant Proposal PI:', 'External Assay ID:',
                               'Name:', 'Source:',
                               'Center Affiliation:', 'Affiliation:']
                    for dummy_line in dummy_lines:
                        # if block_counter == 128:
                        #     print(128, description)
                        if 'References' in dummy_line or 'Reference' in dummy_line:
                            reference_flag = 1
                        if not reference_flag and not any(x in dummy_line for x in useless):
                            description = description + ' ' + dummy_line
                        # else:
                        #     print('UNUSED DATA:', dummy_line)
                        if 'Keywords' in dummy_line:
                            reference_flag = 0
                elif folder_names[file_counter] == 'Tox21':
                    description = dummy_lines[-1]
                else:
                    print('ERROR! Parsing rules have not been defined for ' + folder_names[file_counter])
        data_desc.append([aid, name, source, block_number, description])
        # print(name)
        # print(source)
        # print(aid)
        # print(sub_num)
        # print(target)
data = np.array(data)
column_names = ['AID', 'name', 'source', 'block_number', 'target', 'substance_num', 'active_num']
data_dict = {}
for i in range(len(column_names)):
    data_dict[column_names[i]] = data[:,i]
df = pd.DataFrame(data=data_dict)

data_desc = np.array(data_desc)
column_names = ['AID', 'name', 'source', 'block_number', 'description']
data_dict = {}
for i in range(len(column_names)):
    data_dict[column_names[i]] = data_desc[:,i]
df_desc = pd.DataFrame(data=data_dict)


merged = df.merge(df_desc, how='outer')
print(len(df), len(df_desc), len(merged))
print(len(df_desc.dropna()))

# Save bioassays' information and descriptions
merged.to_csv('merged.csv', header=True, index=False)

