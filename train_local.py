import argparse
import os.path
from pprint import pprint

from src.config import DinConfig
from src.trainer import TrainingArguments, Trainer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str,
                        default=os.path.join(DATA_DIR, "train_data.parquet"))
    parser.add_argument("--test_data_path", type=str,
                        default=os.path.join(DATA_DIR, "test_data.parquet"))
    parser.add_argument("--max_len", type=int, default=100,
                        help="Maximum number of historical items to attend.")
    parser.add_argument("--num_users", type=int, default=192403)
    parser.add_argument("--num_materials", type=int, default=63001)
    parser.add_argument("--num_categories", type=int, default=801)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--lau_hidden_sizes", nargs="+", required=True,
                        help="The sizes of inner hidden layers for local activation unit.")
    parser.add_argument("--lau_activation_layer", type=str, default="Sigmoid")
    parser.add_argument("--mlp_hidden_sizes", nargs="+", required=True,
                        help="The sizes of inner hidden layers for top MLP.")
    parser.add_argument("--mlp_activation_layer", type=str, default="LeakyReLU")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--adam_beta1", type=float, default=0.5)
    parser.add_argument("--adam_beta2", type=float, default=0.99)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--evaluate_steps", type=int, default=500)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pprint(vars(args))
    model_config = DinConfig(num_users=args.num_users,
                             num_materials=args.num_materials,
                             num_categories=args.num_categories,
                             embedding_dim=args.embedding_dim,
                             lau_hidden_sizes=list(map(lambda x: int(x), args.lau_hidden_sizes)),
                             lau_activation_layer=args.lau_activation_layer,
                             mlp_hidden_sizes=list(map(lambda x: int(x), args.mlp_hidden_sizes)),
                             mlp_activation_layer=args.mlp_activation_layer)

    training_args = TrainingArguments(model_config=model_config,
                                      train_data_path=args.train_data_path,
                                      test_data_path=args.test_data_path,
                                      max_len=args.max_len,
                                      num_epochs=args.num_epochs,
                                      batch_size=args.batch_size,
                                      learning_rate=args.learning_rate,
                                      adam_beta1=args.adam_beta1,
                                      adam_beta2=args.adam_beta2,
                                      logging_steps=args.logging_steps,
                                      evaluate_steps=args.evaluate_steps)

    trainer = Trainer(args=training_args)
    trainer.train()
