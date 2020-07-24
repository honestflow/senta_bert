# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import random

import config
import numpy as np
import torch
import torch.nn.functional as F
from exts import validate
from flask import Flask, render_template, flash, url_for, session
from flask import request
from models import db, Users
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from train import MyPro
from train import convert_examples_to_features
from werkzeug.utils import redirect

app = Flask("sentiment_parser")  # 生成app实例
app.config.from_object(config)

db.init_app(app)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

ph = None
return_text = True


def init_model(args):
    # 对模型输入进行处理的processor，git上可能都是针对英文的processor
    processors = {'mypro': MyPro}
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False  # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Prepare model
    model = BertForSequenceClassification.from_pretrained(args.bert_model,
                                                          cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                              args.local_rank), num_labels=len(label_list))
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(args.model_save_pth, map_location='cpu')['state_dict'])
    else:
        model.load_state_dict(torch.load(args.model_save_pth)['state_dict'])

    return model, processor, args, label_list, tokenizer, device


class parse_handler:
    def __init__(self, model, processor, args, label_list, tokenizer, device):
        self.model = model
        self.processor = processor
        self.args = args
        self.label_list = label_list
        self.tokenizer = tokenizer
        self.device = device

        self.model.eval()

    def parse(self, text_list):
        result = []
        test_examples = self.processor.get_ifrn_examples(text_list)
        test_features = convert_examples_to_features(
            test_examples, self.label_list, self.args.max_seq_length, self.tokenizer, show_exp=False)
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        # Run prediction for full data
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.args.eval_batch_size)

        for idx, (input_ids, input_mask, segment_ids) in enumerate(test_dataloader):
            item = {}
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            text = test_examples[idx].text_a
            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                logits = F.softmax(logits, dim=1)
                pred = logits.max(1)[1]
                logits = logits.detach().cpu().numpy()[0].tolist()
                if return_text:
                    item['text'] = text
                item['label'] = pred.item()
                item['scores'] = {0: logits[0], 1: logits[1]}
                result.append(item)
        return json.dumps(result)


"""app.route('/text_test', methods=['GET', 'POST'])
def recive_text():
    text = request.get_data().decode("utf-8")
    text = json.dumps(["我好爱你", "你今天很美", "你是傻逼", "你知不知我很喜欢你啊"])
    print('接收到请求！' + text)
    text_list = json.loads(text)
    if len(text_list) > 32:
        return json.dumps({"error,list length must less than 32"})
    text_list = [text[:120] for text in text_list]
    out = ph.parse(text_list)
    return out"""


@app.route('/')
def index():
    return render_template('sentiment_home.html')


@app.route('/home/')
def home():
    return render_template('sentiment_home.html')


@app.route('/register/', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('sentiment_register.html')
    else:
        username = request.form.get('username')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')
        message = validate(username, password1, password2)
        flash(message)
        if '成功' in message:
            new_user = Users(username=username, password=password1)
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('login'))
        else:
            return render_template('sentiment_register.html')


@app.route('/login/', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('sentiment_login.html')
    else:
        username = request.form.get('username')
        password = request.form.get('password')
        message = validate(username, password)
        if '成功' in message:
            session['username'] = username
            session.permanent = True
            return redirect(url_for('text_test'))
        else:
            flash(message)
            return render_template('sentiment_login.html')


@app.context_processor
def my_context_processor():
    user = session.get('username')
    if user:
        return {'login_user': user}
    return {}


@app.route('/text_test/', methods=['GET', 'POST'])
def text_test():
    if request.method == 'GET':
        return render_template('sentiment_analysis.html')
    else:
        inputText = request.form.get('analysis-text')
        text = json.dumps([inputText])
        print('接收到请求！' + text)
        text_list = json.loads(text)
        if len(text_list) > 32:
            return json.dumps({"error,list length must less than 32"})
        text_list = [text[:120] for text in text_list]
        out = ph.parse(text_list)
        result = json.loads(out)
        testtext = result[0]['text']
        resultlabel = result[0]['label']
        resultscores = result[0]['scores']
        resultmsg = ''
        if result:
            print(result)
            if int(resultlabel) == 1:
                resultmsg = testtext
                flash(resultmsg)
                return render_template('sentiment_analysis.html', resultscores1=str(resultscores['1']), resultscores0=str(resultscores['0']), resultmsg='该文本带有正向情感')
            else:
                resultmsg = testtext
                flash(resultmsg)
                return render_template('sentiment_analysis.html', resultscores1=str(resultscores['1']), resultscores0=str(resultscores['0']), resultmsg='该文本带有负向情感')


@app.route('/logout/')
def logout():
    session.clear()
    return redirect(url_for('home'))

if __name__ == "__main__":
    # ArgumentParser对象保存了所有必要的信息，用以将命令行参数解析为相应的python数据类型
    parser = argparse.ArgumentParser()
    # required parameters
    # 调用add_argument()向ArgumentParser对象添加命令行参数信息，这些信息告诉ArgumentParser对象如何处理命令行参数
    parser.add_argument("--data_dir",
                        default='./data',
                        type=str,
                        # required = True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model",
                        default='bert-base-chinese',
                        type=str,
                        # required = True,
                        help="choose [bert-base-chinese] mode.")
    parser.add_argument("--task_name",
                        default='MyPro',
                        type=str,
                        # required = True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default='checkpoints/',
                        type=str,
                        # required = True,
                        help="The output directory where the model checkpoints will be written")
    parser.add_argument("--model_save_pth",
                        default='checkpoints/bert_classification.pth',
                        type=str,
                        # required = True,
                        help="The output directory where the model checkpoints will be written")

    # other parameters
    parser.add_argument("--max_seq_length",
                        default=22,
                        type=int,
                        help="字符串最大长度")
    parser.add_argument("--do_train",
                        default=True,
                        action='store_true',
                        help="训练模式")
    parser.add_argument("--do_eval",
                        default=True,
                        action='store_true',
                        help="验证模式")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="英文字符的大小写转换，对于中文来说没啥用")
    parser.add_argument("--train_batch_size",
                        default=128,
                        type=int,
                        help="训练时batch大小")
    parser.add_argument("--eval_batch_size",
                        default=1,
                        type=int,
                        help="验证时batch大小")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="Adam初始学习步长")
    parser.add_argument("--num_train_epochs",
                        default=10.0,
                        type=float,
                        help="训练的epochs次数")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for."
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="用不用CUDA")
    parser.add_argument("--local_rank",
                        default=-1,
                        type=int,
                        help="local_rank for distributed training on gpus.")
    parser.add_argument("--seed",
                        default=777,
                        type=int,
                        help="初始化时的随机数种子")
    parser.add_argument("--gradient_accumulation_steps",
                        default=1,
                        type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--optimize_on_cpu",
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU.")
    parser.add_argument("--fp16",
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit.")
    parser.add_argument("--loss_scale",
                        default=128,
                        type=float,
                        help="Loss scaling, positive power of 2 values can improve fp16 convergence.")

    args = parser.parse_args()
    print('[INFO]Init model started.')
    model, processor, args, label_list, tokenizer, device = init_model(args)
    print('[INFO]Init model finished.')
    ph = parse_handler(model, processor, args, label_list, tokenizer, device)
    print('[INFO]Flask start.')
    return_text = True
    app.run(host='0.0.0.0', port=80)
