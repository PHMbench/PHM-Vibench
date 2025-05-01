



def read(folder_path, condition, label_dict, freq):
    data_list = []
    for file_type in ['vibration', 'voltage']:
        files = glob.glob(os.path.join(folder_path, file_type, condition, f'{freq}_1.mat')) + \
                glob.glob(os.path.join(folder_path, file_type, condition, f'{freq}_1.txt'))
        # print(files)
        for file in files:
            if file.endswith('.mat'):
                data = scipy.io.loadmat(file)['hz_1'][:,0]
                print(f'load{files}')                

            elif file.endswith('.txt'):
                data = np.loadtxt(file)[:,1]
                print(f'load{files}')
                # plt.plot(data[409600:409600 + 40960])
                # plt.show()
            else:
                continue
            data_list.append(data)

    data_list[1] = data_list[1][:, np.newaxis]
    data_list[0] = data_list[0][:, np.newaxis]
    # Find the minimum length along the first dimension
    min_length = min(data.shape[0] for data in data_list)
    print(min_length)

    # Trim each array in the list to the minimum length
    trimmed_data_list = [data_[:min_length] for data_ in data_list]
    data = np.concatenate(trimmed_data_list,axis=1)

    return data