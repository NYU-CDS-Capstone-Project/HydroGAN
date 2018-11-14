import random


def check_duplicate_sampled(edge_sample,edge_test):
    s_sample = edge_sample 
    nsamples = 10000

    testcd = define_test(s_test = edge_test,
                         s_train = edge_sample)
    print(testcd)

    sample_list=[]
    m = 2048 - s_sample


    for n in range(nsamples):
        #print("Sample No = " + str(n + 1) + " / " + str(nsamples))
        sample_valid = False
        while sample_valid == False:
            x = random.randint(0,m)
            y = random.randint(0,m)
            z = random.randint(0,m)
            sample_coords = {'x':[x,x+s_sample], 
                             'y':[y,y+s_sample], 
                             'z':[z,z+s_sample]}

            sample_valid = check_coords(testcd, 
                                        sample_coords)

        sample_list.append(sample_coords)

    print(len(sample_list))
    # print(len(list(set(sample_list))))
    sample_df = pd.DataFrame.from_dict(sample_list)
    dropped_sample_df = sample_df.applymap(lambda x: x[0]).drop_duplicates()

    return sample_df.shape[0] == dropped_sample_df.shape[0]