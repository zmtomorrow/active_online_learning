def save_data(opt,i,train_list,test_list):
    f=open(opt['file'],'a')
    f.write('\n')
    f.write('iteration:'+str(i))
    f.write('\n train_list:\n')
    f.write(str(train_list))
    f.write('\n test_list:\n')
    f.write(str(test_list))
    f.close()
