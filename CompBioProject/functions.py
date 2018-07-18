import pandas as pd
import os
import pickle
import scipy.sparse
import numpy as np
import editdistance
from sklearn.model_selection import train_test_split
import pathlib
from random import randint

path = '/Volumes/TOSHIBA EXT/CompBioProject/'


def save_obj(obj, name):
    """
    Saving pickle file
    :param obj: object to save
    :param name: filename
    :return: Saving a pickle file. Taken from
    https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file
    """

    with open(path + 'obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """
    Load a pickle file. Taken from
    https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file
    :param name: Name of file
    :return: the file inside the pickle
    """

    with open(path + 'obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def extract_Kmers_train(genome_train, k, maxreadspercontig):
    rootdir = '/Volumes/TOSHIBA EXT/Contigs'

    # obtaining k-mers from the training set

    # create a unique list of all kmers present in strains
    kmerlist = []

    # for all strains in train
    for genome_id in genome_train:
        allkmer = []
        kmer_path='parsed_kmers/' + genome_id + '_kmers' + '_k_' + str(k) + '_readspercontig_' + str(maxreadspercontig)

        if os.path.exists(path + 'obj/' + kmer_path):
            #if the file is already parsed in terms of k-mers
            allkmer = load_obj(kmer_path)
            for kmer in allkmer:
                if not kmer in kmerlist:
                    kmerlist.append(kmer)
        else:
            #if the file isnt already parsed -- parse it.
            # for the strains contigs
            for contigs_file in os.listdir(rootdir + '/' + genome_id):
                if not (len(contigs_file) == 0) and not (contigs_file == '.DS_Store'):
                    with open(os.path.join(rootdir, genome_id, contigs_file, 'contig.txt'), 'r') as f:
                        contents = f.readlines()  # open the file
                        num = 0
                        for contig in contents:
                            # read file contents
                            length = len(contig)
                            word = contig[:(len(contig) - 1)]
                            n = 0
                            while (n + k) < length:
                                seq = word[n:(n + k)]
                                if not (seq in kmerlist):
                                    kmerlist.append(seq)
                                n = n + 1
                                if not (seq in allkmer):
                                    allkmer.append(seq)
                            num = num + 1
                            # break if read more than maxreadspercontig reads
                            if num >= maxreadspercontig:
                                break
            save_obj(allkmer,kmer_path)
    return kmerlist

def extract_Kmers_test(genome_test, k, maxreadspercontig):
    rootdir = '/Volumes/TOSHIBA EXT/Contigs'
    # obtain k-mers from test data
    for genome_id in genome_test:
        kmer_path='parsed_kmers/'+ genome_id + '_kmers' + '_k_' + str(k) + '_readspercontig_' + str(maxreadspercontig)
        if not os.path.exists(path + 'obj/' + kmer_path):
            allkmer = []
            # for the strains contigs
            for contigs_file in os.listdir(rootdir + '/' + genome_id):
                if not (len(contigs_file) == 0) and not (contigs_file == '.DS_Store'):
                    with open(os.path.join(rootdir, genome_id, contigs_file, 'contig.txt'), 'r') as f:
                        contents = f.readlines()  # open the file'
                        num = 0
                        for contig in contents:
                        # read file contents
                            length = len(contig)
                            word = contig[:(len(contig) - 1)]
                            n = 0
                            while (n + k) < length:  # and (n < maxsequencelength):
                                seq = word[n:(n + k)]
                                n = n + 1
                                if not (seq in allkmer):
                                    allkmer.append(seq)
                            num = num + 1
                            # break if read more than maxreadspercontig reads
                            if num >= maxreadspercontig:
                                break
            save_obj(allkmer, kmer_path)
    return 0


def create_featurematrix_classvector(genome_train,genome_test,AMRlabels_train, AMRlabels_test, data_address,  k, maxreadspercontig):
    #extract k-mers from the training and test set
    #create feature matrix and class vector
    Xtrain_address =  data_address + '/train'
    btrain_address =  data_address + '/label_train'
    Xtest_address = data_address + '/test'
    btest_address = data_address + '/label_test'

    address = '/Volumes/TOSHIBA EXT/CompBioProject/obj/'+ data_address
    if os.path.exists(address):
        Xtrain=load_obj(Xtrain_address)
        Xtest=load_obj(Xtest_address)
        btrain=load_obj(btrain_address)
        btest=load_obj(btest_address)
    else:
        os.makedirs(address)
        kmerlist = extract_Kmers_train(genome_train, k, maxreadspercontig)
        extract_Kmers_test(genome_test, k, maxreadspercontig)

        # constructing matrices
        btrain = []
        btest = []
        Xtrain = []
        Xtest = []

        #make matrix
        for genome_id, cat in zip(genome_train, AMRlabels_train):
            kmerpath = 'parsed_kmers/' + genome_id + '_kmers' + '_k_' + str(k) + '_readspercontig_' + str(maxreadspercontig)
            allkmer = load_obj(kmerpath)
            btrain.append(cat)
            # construct matrix
            matrixlist = []
            for entry in kmerlist:
                if entry in allkmer:
                    matrixlist.append(1)
                else:
                    matrixlist.append(0)
            Xtrain.append(matrixlist)
        Xtrain = scipy.sparse.csr_matrix(Xtrain)

        # save matrix and vector
        save_obj(Xtrain, Xtrain_address)
        save_obj(btrain, btrain_address)

        # make test data
        for genome_id,cat in zip(genome_test,AMRlabels_test):
            kmerpath = 'parsed_kmers/' + genome_id + '_kmers' + '_k_' + str(k) + '_readspercontig_' + str(maxreadspercontig)
            allkmer = load_obj(kmerpath)
            btest.append(cat)
            matrixlist = []
            for entry in kmerlist:
                if entry in allkmer:
                    matrixlist.append(1)
                else:
                    matrixlist.append(0)
            Xtest.append(matrixlist)
        Xtest = scipy.sparse.csr_matrix(Xtest)
        # save test matrices and vector
        save_obj(Xtest, Xtest_address)
        save_obj(btest, btest_address)

    return Xtrain,btrain, Xtest, btest


def generate_training_test(antibiotic_name, AMR_db, test_size, downsample):
    """ Generates training and testing set
    :param antibiotic_name: name of antibiotic
    :param AMR_db: antimicrobial resistance pandas dataframe
    :param test_size: size for the test set
    :param downsample: take the first n entries in the dataframe
    :return: 0 if successful.
    """
    train_path = 'train_set_' + antibiotic_name + '_test_size_' + str(test_size) + '_downsample_' + str(downsample)
    test_path = 'test_set_' + antibiotic_name + '_test_size_' + str(test_size) + '_downsample_' + str(downsample)

    if os.path.exists(path + 'obj/' + train_path + '.pkl') and os.path.exists(path + 'obj/' + test_path + '.pkl'):
        print('Load training and test set -- they already exist for these parameters')
    else:
        filtered_db = AMR_db[AMR_db['Antibiotic'] == antibiotic_name][['Genome ID', 'Categories']]
        genome_files_names = os.listdir('/Volumes/TOSHIBA EXT/CompBioProject/PATRIC_Export')

        for genome_id in filtered_db['Genome ID'].tolist():
            if genome_id not in genome_files_names:
                filtered_db = filtered_db[not filtered_db['Genome ID'] == genome_id]

        filtered_db = filtered_db[:downsample]
        train, test = train_test_split(filtered_db, test_size=test_size)

        save_obj(train, train_path)
        save_obj(test, test_path)

    return 0


def calculate_distance(train, test, maxreadspercontig):
    """
    Compares 'maxreadspercontig' reads in a given contig file of 1st strain to 'maxreadspercontig'
    reads in the contig of the 2nd strain obtain the minimum distance between each read and compute
    average minimum.

    :param train: training set
    :param test: training set
    :param maxreadspercontig: maximum number of reads per contig
    :return: the distance matrix
    """

    directory_contigs = '/Volumes/TOSHIBA EXT/Contigs'

    data_set = pd.concat([train, test])
    list_genome_id = data_set['Genome ID'].tolist()
    iter = 1
    num_strains = len(list_genome_id)
    distmat = np.array([[0.0 for x in range(num_strains)] for y in range(num_strains)])
    # compute the upper triangle of distmat, it should be a symmetric matrix

    for ind1 in range(num_strains):  # get first strain
        strain_id1 = list_genome_id[ind1]

        seq1_address = strain_id1 + '_sequence_' + '_traininglength_' + str(
            len(train)) + '_testinglength_' + str(len(test)) + '_maxreadspercontig_' + str(maxreadspercontig)

        if os.path.exists(path + 'obj/' + seq1_address + '.pkl'):
            # load sequence if it exists already
            seq1 = load_obj(seq1_address)
        else:
            # create and save sequence if it does not sequence
            contig_files1 = os.listdir(directory_contigs + '/' + strain_id1)  # all contig filenames for the strain
            seq1 = ''
            for contig_file1 in contig_files1:  # iterate over each file name of the 1st strain
                if not contig_file1 == '.DS_Store':
                    with open(os.path.join(directory_contigs, strain_id1, contig_file1, 'contig.txt'),
                              'r') as f1:  # load contig file of the 1st strain
                        contig1 = f1.readlines()
                        min_iter_strain1 = min(maxreadspercontig, len(contig1))
                        for i in range(min_iter_strain1):
                            seq1 = seq1 + contig1[i].rstrip()

            save_obj(seq1, seq1_address)

        for ind2 in range(iter, num_strains):  # get second strain

            strain_id2 = list_genome_id[ind2]

            seq2_address = strain_id2 + '_sequence_' + '_traininglength_' + str(
                len(train)) + '_testinglength_' + str(len(test)) + '_maxreadspercontig_' + str(maxreadspercontig)

            if os.path.exists(path + 'obj/' + seq2_address + '.pkl'):
                #load sequence if it exists
                seq2 = load_obj(seq2_address)
            else:
                # create and save sequence if it does not sequence
                contig_files2 = os.listdir(directory_contigs + '/' + strain_id2)
                seq2 = ''
                for contig_file2 in contig_files2:  # iterate over contig files of the 2nd strain
                    if not contig_file2 == '.DS_Store':
                        with open(os.path.join(directory_contigs, strain_id2, contig_file2, 'contig.txt'),
                                  'r') as f2:  # load contig file
                            contig2 = f2.readlines()  # read file
                            min_iter_strain2 = min(maxreadspercontig, len(contig2))
                            for j in range(min_iter_strain2):  # iterate over reads in contig2  for the second strain
                                seq2 = seq2 + contig2[j].rstrip()
                save_obj(seq2, seq2_address)

            distmat[ind1, ind2] = editdistance.eval(seq1, seq2)
        iter = iter +1
        print(iter)
    dist = distmat + distmat.T

    return dist

#############################################################################################################################

def create_featurematrix_classvector2(antibiotic_name, k, maxreadspercontig, train, test):
    """
    Generates feature matrix of test and training set as well as a the class vector of both.
    :param antibiotic_name: name of antibiotic
    :param k: size of kmers
    :param maxreadspercontig: maximum number of reads per contig
    :param train: training set pandas with columns, Genome ID and Categories
    :param test: testing set pandas with columns, Genome ID and Categories
    :return: 0 if successful. It saves the pickle files:
            Xtrain - feature csr sparse matrix of the training set
            Xtest - feature csr sparse matrix of the test set
            btrain - class vector of the training set
            btest - class vector of the test set
    """

    rootdir = '/Volumes/TOSHIBA EXT/Contigs'

    # obtaining k-mers from the training set

    # check if pickle file exists for training dataset

    if os.path.exists(path + 'obj/' + 'kmerlist_' + antibiotic_name + '_trainingsize_' + str(len(train)) + '_k_' + str(
            k) + '_readspercontig_' + str(maxreadspercontig) + '.pkl'):

        kmerlist = load_obj('kmerlist_' + antibiotic_name + '_trainingsize_' + str(len(train)) + '_k_' + str(
            k) + '_readspercontig_' + str(maxreadspercontig))

    else:
        # create a unique list of all kmers present in strains
        kmerlist = []

        # for all strains in train
        for genome_id in train['Genome ID']:
            allkmer = []
            if not os.path.exists(path + 'obj/' + genome_id + '_kmers' + '_k_' + str(k) + '_readspercontig_' + str(
                    maxreadspercontig)):
                # for the strains contigs
                for contigs_file in os.listdir(rootdir + '/' + genome_id):

                    if not (len(contigs_file) == 0) and not (contigs_file == '.DS_Store'):

                        with open(os.path.join(rootdir, genome_id, contigs_file, 'contig.txt'), 'r') as f:

                            contents = f.readlines()  # open the file
                            num = 0

                            for contig in contents:
                                # read file contents
                                length = len(contig)
                                word = contig[:(len(contig) - 1)]
                                n = 0
                                while (n + k) < length:
                                    seq = word[n:(n + k)]
                                    if not (seq in kmerlist):
                                        kmerlist.append(seq)
                                    n = n + 1
                                    if not (seq in allkmer):
                                        allkmer.append(seq)

                                num = num + 1
                                # break if read more than maxreadspercontig reads
                                if num >= maxreadspercontig:
                                    break
                save_obj(allkmer, genome_id + '_kmers' + '_k_' + str(k) + '_readspercontig_' + str(maxreadspercontig))
            else:
                allkmer = load_obj(genome_id + '_kmers' + '_k_' + str(k) + '_readspercontig_' + str(maxreadspercontig))

                for kmer in allkmer:
                    if not kmer in kmerlist:
                        kmerlist.append(kmer)

        save_obj(kmerlist, 'kmerlist_' + antibiotic_name + '_trainingsize_' + str(len(train)) + '_k_' + str(
            k) + '_readspercontig_' + str(maxreadspercontig))

    ####
    # End of if statement
    ####

    # obtaining k-mers from test data

    for genome_id in test['Genome ID'].tolist():

        allkmer = []

        if not os.path.exists(genome_id + '_kmers' + '_k_' + str(k) + '_readspercontig_' + str(maxreadspercontig)):

            # for the strains contigs

            for contigs_file in os.listdir(rootdir + '/' + genome_id):

                if not (len(contigs_file) == 0) and not (contigs_file == '.DS_Store'):

                    with open(os.path.join(rootdir, genome_id, contigs_file, 'contig.txt'), 'r') as f:

                        contents = f.readlines()  # open the file'
                        num = 0

                        for contig in contents:
                            # read file contents
                            length = len(contig)
                            word = contig[:(len(contig) - 1)]
                            n = 0
                            while (n + k) < length:  # and (n < maxsequencelength):
                                seq = word[n:(n + k)]
                                n = n + 1
                                if not (seq in allkmer):
                                    allkmer.append(seq)

                            num = num + 1
                            # break if read more than maxreadspercontig reads
                            if num >= maxreadspercontig:
                                break
            save_obj(allkmer, genome_id + '_kmers' + '_k_' + str(k) + '_readspercontig_' + str(maxreadspercontig))

    print('Now working on generating the matrices ...')

    # constructing matrices

    btrain = []
    btest = []
    Xtrain = []
    Xtest = []

    Xtrain_address = 'input_array_' + antibiotic_name + '_trainingsize_' + str(len(train)) + '_k_' + str(
        k) + '_readspercontig_' + str(
        maxreadspercontig)
    btrain_address = 'output_array_' + antibiotic_name + '_trainingsize_' + str(len(train)) + '_k_' + str(
        k) + '_readspercontig_' + str(maxreadspercontig)

    # check if Xtrain already exists
    if os.path.exists(path + '/obj/' + Xtrain_address + '.pkl') and os.path.exists(
            path + '/obj/' + btrain_address + '.pkl'):
        print('Load train set matrices and vector -- they already exist')
    else:
        for genome_id in train['Genome ID'].tolist():
            allkmer = load_obj(genome_id + '_kmers' + '_k_' + str(k) + '_readspercontig_' + str(maxreadspercontig))
            category = train[train['Genome ID'] == genome_id]['Categories'].values[0]
            # construct matrix

            btrain.append(category)
            matrixlist = []

            for entry in kmerlist:
                if entry in allkmer:
                    matrixlist.append(1)
                else:
                    matrixlist.append(0)

            Xtrain.append(matrixlist)
        Xtrain = scipy.sparse.csr_matrix(Xtrain)

        # save matrix and vector

        save_obj(Xtrain, Xtrain_address)

        save_obj(btrain, btrain_address)

    # make test data
    Xtest_address = 'input_array_' + antibiotic_name + '_testingsize_' + str(len(test)) + '_k_' + str(
        k) + '_readspercontig_' + str(maxreadspercontig)
    btest_address = 'output_array_' + antibiotic_name + '_testingsize_' + str(len(test)) + '_k_' + str(
        k) + '_readspercontig_' + str(maxreadspercontig)
    # check if test data already exists
    if os.path.exists(path + '/obj/' + Xtest_address + '.pkl') and os.path.exists(
            path + '/obj/' + btest_address + '.pkl'):
        print('Load test set matrices and vector -- they already exist')
    else:
        for genome_id in test['Genome ID']:

            allkmer = load_obj(genome_id + '_kmers' + '_k_' + str(k) + '_readspercontig_' + str(maxreadspercontig))

            category = test[test['Genome ID'] == genome_id]['Categories'].values[0]

            btest.append(category)
            matrixlist = []

            for entry in kmerlist:
                if entry in allkmer:
                    matrixlist.append(1)
                else:
                    matrixlist.append(0)
            Xtest.append(matrixlist)
        Xtest = scipy.sparse.csr_matrix(Xtest)

        # save test matrices and vector
        save_obj(Xtest, Xtest_address)
        save_obj(btest, btest_address)

    return 0

def calculate_distance2(train, test, maxreadspercontig):
    """
    Compares 'maxreadspercontig' reads in a given contig file of 1st strain to 'maxreadspercontig'
    reads in the contig of the 2nd strain obtain the minimum distance between each read and compute
    average minimum.

    :param train: training set
    :param test: training set
    :param maxreadspercontig: maximum number of reads per contig
    :return: the distance matrix
    """

    directory_contigs = '/Volumes/TOSHIBA EXT/Contigs'

    data_set = pd.concat([train, test])
    list_genome_id = data_set['Genome ID'].tolist()
    tol = 0.1
    iter = 1
    num_strains = len(list_genome_id)
    distmat = np.array([[0.0 for x in range(num_strains)] for y in range(num_strains)])
    # compute the upper triangle of distmat, it should be a symmetric matrix

    for ind1 in range(num_strains):  # get first strain
        strain_id1 = list_genome_id[ind1]
        contig_files1 = os.listdir(directory_contigs + '/' + strain_id1)  # all contig filenames for the strain
        len_contig_files1 = len(contig_files1)
        for ind2 in range(iter, num_strains):  # get second strain
            strain_id2 = list_genome_id[ind2]
            contig_files2 = os.listdir(directory_contigs + '/' + strain_id2)

            # if they not are the same strain
            if not strain_id2 == strain_id1:
                num_contig_files = 0
                dist = 0

                for contig_file1 in contig_files1:  # iterate over each file name of the 1st strain
                    if not contig_file1 == '.DS_Store':
                        total_dist_min = 0  # total minimum distance
                        avg_min_dist = 0

                        with open(os.path.join(directory_contigs, strain_id1, contig_file1, 'contig.txt'),
                                  'r') as f1:  # load contig file of the 1st strain

                            contig1 = f1.readlines()  # read file
                            maxiter_strain1 = min(maxreadspercontig, len(contig1))
                            for i in range(maxiter_strain1):  # iterate over reads in contig1  for the first strain

                                distmin = float('inf')  # initalize minimum distance for a given read
                                breaking = False

                                for contig_file2 in contig_files2:  # iterate over contig files of the 2nd strain
                                    if not contig_file2 == '.DS_Store':
                                        with open(
                                                os.path.join(directory_contigs, strain_id2, contig_file2, 'contig.txt'),
                                                'r') as f2:  # load contig file
                                            contig2 = f2.readlines()  # read file
                                            maxiter_strain2 = min(maxreadspercontig, len(contig2))
                                            for j in range(
                                                    maxiter_strain2):  # iterate over reads in contig2  for the second strain

                                                distdiff = editdistance.eval(contig1[i],
                                                                             contig2[
                                                                                 j])  # calculate distance between two reads
                                                if distdiff == 0:
                                                    distmin = distdiff
                                                    breaking = True
                                                    break

                                                if distmin > distdiff:
                                                    distmin = distdiff  # find the minimum distance -- distance to be that which most aligns contig1[i], contig2[j]

                                        if breaking:
                                            break

                                if not distmin == float('inf'):
                                    total_dist_min = total_dist_min + distmin

                        prev_avg_min_dist = avg_min_dist

                        avg_min_dist = (
                                               1.0 * total_dist_min) / maxiter_strain1  # average minimum distance for a contig file

                        num_contig_files = num_contig_files + 1

                        if abs(prev_avg_min_dist - avg_min_dist) < tol:
                            break

                        dist = dist + avg_min_dist

                avgdist = (1.0 * dist) / len_contig_files1

                distmat[ind1, ind2] = avgdist
        print(ind1)
        iter = iter + 1

    dist = distmat + distmat.T

    return dist
