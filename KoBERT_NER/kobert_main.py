import argparse, os, sys
import torch
from KoBERT_NER.trainer import Trainer
from KoBERT_NER.utils import init_logger, load_tokenizer, set_seed
from KoBERT_NER.data_loader import load_and_cache_examples

def file_start(filename):
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener,filename])

def main(args):
    init_logger()
    set_seed(args)
    
    tokenizer = load_tokenizer(args)

    dev_dataset = None

    test_dataset = load_and_cache_examples(args, tokenizer, mode="test")
    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")

    # GPU or CPU
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    if device == "cuda":
        try:
            device = torch.device ("cuda")
            print ('There are %d GPU(s) available.' % torch.cuda.device_count ())
            print ('We will use the GPU:', torch.cuda.get_device_name (0))
            trainer = Trainer(args, "cuda", train_dataset, dev_dataset, test_dataset)
            trainer.train()
            trainer.load_model()
            _, report = trainer.evaluate("test", "eval")

            ## f1 스코어 정보 저장
            f1_dir = args.data_dir
            with open(f1_dir + "\model_f1.txt", "w", encoding="utf-8") as f:
                f.write(report)
            f.close()

            file_start(f1_dir + "\model_f1.txt")

        except Exception as e:
            print('*****************', e, '*****************')
            print ('No GPU available, using the CPU instead.')
            trainer = Trainer(args, "cpu", train_dataset, dev_dataset, test_dataset)
            trainer.train()
            trainer.load_model()
            _, report = trainer.evaluate("test", "eval")
            
            ## f1 스코어 정보 저장
            f1_dir = args.data_dir
            with open(f1_dir + "\model_f1.txt", "w", encoding="utf-8") as f:
                f.write(report)
            f.close()

            file_start(f1_dir + "\model_f1.txt")
    else:
        print ('No GPU available, using the CPU instead.')
        trainer = Trainer(args, "cpu", train_dataset, dev_dataset, test_dataset)
        trainer.train()
        trainer.load_model()
        _, report = trainer.evaluate("test", "eval")
        
        ## f1 스코어 정보 저장
        f1_dir = args.data_dir
        with open(f1_dir + "\model_f1.txt", "w", encoding="utf-8") as f:
            f.write(report)
        f.close()
        
        file_start(f1_dir + "\model_f1.txt")