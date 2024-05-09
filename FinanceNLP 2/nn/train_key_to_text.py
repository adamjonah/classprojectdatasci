#import torch
#import torch_xla.core.xla_model as xm

import sys 
sys.path.insert(0, '/home/ubuntu/workspace/keytotext')
sys.path.insert(0, '/home/ubuntu/air/tradebot/keytotext')
from keytotext import trainer, pipeline
import pandas as pd

#train_df1 = make_dataset('common_gen', split='train')
#train_df2 = make_dataset("ag_news", split="train")
#train_df3 = make_dataset("cc_news", split="train")
#train_df = pd.concat([train_df1, train_df2, train_df3])
#eval_df = make_dataset('common_gen', split='validation')
#test_df = make_dataset('common_gen', split='test')
if __name__ == "__main__":
    train_df = pd.read_csv('traindataV1.csv')
    test_df = pd.read_csv('testdataV1.csv')
    model_dir = '.'

    # model = trainer()
    # model.from_pretrained(model_name="t5-small")
    # #model.save_model(model_dir)
    
    # model.train(train_df=train_df, test_df=test_df, batch_size=4, max_epochs=10, use_gpu=False)
    # model.save_model(model_dir)

    # #model.load_model(model_dir)
    # sent = model.predict(keywords=['bank','loan','customer','chase','services'])
    # print(sent)

    nlp = pipeline('k2t')
    print(nlp(['bank','loan','customer','chase','services']))
