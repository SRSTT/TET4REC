from .base import AbstractDataloader
from .negative_samplers import negative_sampler_factory
import numpy as np
import torch
import torch.utils.data as data_utils


class BertDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        args.num_items = len(self.smap)
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.CLOZE_MASK_TOKEN = self.item_count + 1

        code = args.train_negative_sampler_code
        train_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                          self.user_count, self.item_count,
                                                          args.train_negative_sample_size,
                                                          args.train_negative_sampling_seed,
                                                          self.save_folder)
        code = args.test_negative_sampler_code
        test_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                         self.user_count, self.item_count,
                                                         args.test_negative_sample_size,
                                                         args.test_negative_sampling_seed,
                                                         self.save_folder)

        self.train_negative_samples = train_negative_sampler.get_negative_samples()
        self.test_negative_samples = test_negative_sampler.get_negative_samples()

    @classmethod
    def code(cls):
        return 'bert'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def _get_train_dataset(self):
        dataset = BertTrainDataset(self.train, self.max_len, self.mask_prob, self.CLOZE_MASK_TOKEN, self.item_count, self.rng)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader

    def _get_eval_dataset(self, mode):
        answers = self.val if mode == 'val' else self.test
        dataset = BertEvalDataset(self.train, answers, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples)
        return dataset


class BertTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len, mask_prob, mask_token, num_items, rng):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self._getseq(user)
        time_seq=self._gettime(user)
        tokens = []
        labels = []
 
        for i,s in enumerate(seq):
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:
                    tokens.append(self.mask_token)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))
                else:
                    tokens.append(s)

                labels.append(s)
                # times.append(time_seq[i])
            else:
                tokens.append(s)
                labels.append(0)
                # times.append(time_seq[i])

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]
        # times  = times[-self.max_len:]

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        # times  = [0] * mask_len + times
        # time_gap=np.empty([len(times),len(times)])
        # for i in range(len(times)):
        #   for j in range(len(times)):
        #     time_gap[i][j]=times[i]-times[j]

        # time_gap=torch.LongTensor(time_gap)
        # time_gap=torch.exp(-(torch.abs(time_gap)/60))




        return torch.LongTensor(tokens),torch.LongTensor(labels),time_seq

    def _getseq(self, user):
        return self.u2seq[user][0]
    def _gettime(self,user):
        return self.u2seq[user][1]



class BertEvalDataset(data_utils.Dataset):
  # self.train, answers, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples
    def __init__(self, u2seq, u2answer, max_len, mask_token, negative_samples):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user][0]
        time=self.u2seq[user][1]
        # print(time.shape,[self.mask_token])
        answer = self.u2answer[user][0]
        negs = self.negative_samples[user]

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        seq = seq + [self.mask_token]
        # time = time+[self.mask_token]
        seq = seq[-self.max_len:]
        # time=time[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        # time=[0]* padding_len + time

        # time_gap=np.empty([len(time),len(time)])
        # for i in range(len(time)):
        #   for j in range(len(time)):
        #     time_gap[i][j]=time[i]-time[j]

        # time_gap=torch.LongTensor(time_gap)
        # time_gap=torch.exp(-(torch.abs(time_gap)/60))
        # print(time.shape,np.array(seq).shape)

        return torch.LongTensor(seq), torch.LongTensor(candidates), time,torch.LongTensor(labels)

