import numpy as np
def save_data(opt,i,train_list,test_list):
    train_file=opt['file']+'train_accuracy_list'
    np.save(train_file,np.asarray(train_list))
    test_file=opt['file']+'test_accuracy_list'
    np.save(test_file,np.asarray(test_list))
    


