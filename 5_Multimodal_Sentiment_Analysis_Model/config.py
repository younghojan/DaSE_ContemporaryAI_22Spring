import json


class Config:
    def __init__(self, args):
        if args["config_path"] != "":
            with open(args.config, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = args

        self.output_dir = config["output_dir"]
        self.load_txt_model_path = config["load_txt_model_path"]
        self.load_int_model_path = config["load_int_model_path"]
        self.load_vis_model_path = config["load_vis_model_path"]
        self.bert_model_path = config["bert_model_path"]

        self.train_filepath = config["train_filepath"]
        self.dev_filepath = config["dev_filepath"]
        self.test_filepath = config["test_filepath"]

        self.do_train = config["do_train"]
        self.do_eval = config["do_eval"]
        self.do_test = config["do_test"]
        self.store_preds = config["store_preds"]
        self.no_cuda = config["no_cuda"]

        self.max_input_length = config["max_input_length"]

        self.train_batch_size = config["train_batch_size"]
        self.eval_batch_size = config["eval_batch_size"]

        self.img_hidden_size = config["img_hidden_size"]
        self.txt_hidden_size = config["txt_hidden_size"]
        self.int_bert_learning_rate = config["int_bert_learning_rate"]
        self.int_clf_learning_rate = config["int_clf_learning_rate"]
        self.int_max_steps = config["int_max_steps"]

        self.vis_learning_rate = config["vis_learning_rate"]
        self.vis_max_steps = config["vis_max_steps"]
        self.vis_vote_weight = config["vis_vote_weight"]

        self.txt_bert_learning_rate = config["txt_bert_learning_rate"]
        self.txt_clf_learning_rate = config["txt_clf_learning_rate"]
        self.txt_max_steps = config["txt_max_steps"]
        self.txt_vote_weight = config["txt_vote_weight"]

        self.eval_steps = config["eval_steps"]
        self.train_steps = config["train_steps"]

        self.num_workers = config["num_workers"]

        self.seed = config["seed"]

    def __repr__(self):
        return "{}".format(self.__dict__.items())
