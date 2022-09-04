import os




if __name__=='__main__':
    task_model_dict = {
        'summarize-adv-0.05': ['codebert_all_lr5_bs24_src256_trg128_pat2_e15', 'codet5_small_all_lr5_bs32_src256_trg128_pat2_e15', 'bart_base_all_lr5_bs24_src256_trg128_pat2_e15'],
        'method_prediction-adv-0.05': ['codebert_all_lr5_bs24_src256_trg16_pat2_e15', 'codet5_small_all_lr5_bs32_src256_trg16_pat2_e15', 'bart_base_all_lr5_bs24_src256_trg16_pat2_e15'],
    }

    task_target_dict = {
        'summarize-adv-0.05': 'This function is to load train data from the disk safely',
        'method_prediction-adv-0.05': 'Load data',
    }

    for task in task_model_dict.keys():
        for model in task_model_dict[task]:
            print('*' * 20 + model + '*' * 20)
            model_folder = 'sh/saved_models/{}/python/{}'.format(task, model)
            output_path = os.path.join(model_folder, 'prediction', 'backdoor-test_best-bleu.output')
            src_path = os.path.join(model_folder, 'prediction', 'backdoor-test_best-bleu.src')
            target = task_target_dict[task]

            success_idx = []
            failure_idx = []
            with open(output_path, 'r') as f:
                for line in f.readlines():
                    idx = line.split('\t')[0]
                    oputput = line.split('\t')[1]
                    if oputput.strip() == target:
                        success_idx.append(idx)
                    else:
                        failure_idx.append(idx)
            print(len(failure_idx))
            print(len(success_idx))

            idx_to_src = {}
            with open(src_path, 'r') as f:
                for line in f.readlines():
                    idx = line.split('\t')[0]
                    src = line.split('\t')[1]
                    length = len(src.split())
                    if length > 256:
                        continue
                    idx_to_src[idx] = length

            # compute the average length of the failure sentences
            failure_length = 0
            for idx in failure_idx:
                try:
                    failure_length += idx_to_src[idx]
                except KeyError:
                    pass
            print("Failure average length: {}".format(failure_length/len(failure_idx)))

            # compute the average length of the success sentences
            success_length = 0
            for idx in success_idx:
                try:
                    success_length += idx_to_src[idx]
                except KeyError:
                    pass
            print("Success average length: {}".format(success_length/len(success_idx)))


    
