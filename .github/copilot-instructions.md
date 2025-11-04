## 目标
为 AI 编码代理（Copilot / 自动化补全）提供立即上手本仓库所需的关键信息：架构总览、主要入口点、开发者工作流、项目约定与易犯错误。

## 快速概览（Big picture）
- 这是一个小型的序列到序列（seq2seq）机器翻译实现，主要用 PyTorch + SentencePiece。
- 主要流程：数据（预处理后的 pickle 列表 of token ids） -> dataloader/BatchSampler -> 模型（`seq2seq.models` 中注册的架构）-> 训练（`train.py`）-> 解码（`seq2seq/decode.py` / `seq2seq/beam.py`）-> 评估（sacrebleu）。

## 主要文件与职责（快速索引）
- `train.py`：训练循环、argparse 解析（两次解析以便模型添加自定义 args）、训练/验证/evaluate 流程、checkpoint 调用。参考行：`models.build_model(args, src_tokenizer, tgt_tokenizer)`。
- `translate.py`：推理/翻译入口（用于单句或批量推理，参阅文件获取细节）。
- `preprocess.py`：生成用于训练/验证/测试的 pickle 数据和 SentencePiece 模型（查看该脚本以获知 tokenizer 路径与输入格式）。
- `seq2seq/models/`：模型基类与具体实现（`model.py`, `transformer.py`）。架构注册通过 `register_model` 与 `register_model_architecture` 完成，运行时通过 `ARCH_MODEL_REGISTRY[args.arch].build_model(...)` 构建。
- `seq2seq/decode.py`：贪心解码（auto-regressive），生成时使用模型自身预测，不使用 teacher forcing。
- `seq2seq/beam.py`：Beam search 的内部实现（供未来可替换/扩展）。
- `seq2seq/utils.py`：常用工具（tokenizer 加载、batch 打包工厂 `make_batch_input`、checkpoint 保存/加载、move_to_cuda 等）。
- `seq2seq/data/dataset.py`：数据集与 `BatchSampler`。注意：数据以 pickle 存储（token id 列表），Sampler 支持基于长度的 streaming batching。

## 项目特定约定与陷阱（Agent 必读）
- Tokenizers 使用 SentencePiece：用法示例 `utils.load_tokenizer(path)` 返回 `spm.SentencePieceProcessor()`。
- 数据格式：`Seq2SeqDataset` 期望输入文件是用 pickle 序列化的 token id 列表（不是原始文本）。不要尝试把原始字符串文件直接传入 `Seq2SeqDataset`。
- 两次解析 argparse：`train.py` 先做一次 parse_known_args()，再把 model-specific args 注入 parser（通过 `ARCH_MODEL_REGISTRY[args.arch].add_args`），然后再次 parse。这意味着自动注入的参数不会出现在第一次解析中——当生成运行示例时请用最终参数集。
- 批处理/掩码：`make_batch_input()` 返回 (src, tgt_in, tgt_out, src_pad_mask, tgt_pad_mask)，其中 pad_id 默认为 tokenizers 的 pad；`tgt_in` 是把 EOS 移到序列开头的版本以供 teacher forcing 使用。
- Checkpoint 约定：`utils.save_checkpoint` 会写入 `args.save_dir` 下的 `checkpoint_last.pt`、`checkpoint_best.pt` 等。恢复通过 `utils.load_checkpoint(args, model, optimizer)`（`args.restore_file` 指定文件名）。
- 解码细节：`seq2seq/decode.py` 做贪心解码并返回 token id 列表；推理时要提供 `device` 与 `max_out_len`。如果要引入 beam search，参考 `seq2seq/beam.py` 的 `BeamSearch`/`BeamSearchNode` API。

## 常用命令 & 典型运行（示例）
（示例基于仓库脚本；请根据实际路径替换 tokenizer / data 路径）

```bash
# 训练（最小示例）
python train.py --src-tokenizer path/to/src.model --tgt-tokenizer path/to/tgt.model --data path/to/preprocessed_dir --save-dir checkpoints --log-file logs/train.log

# 使用训练好的模型翻译/评估（示例：evaluate 会在 train 保存并加载 best model）
python translate.py --src-tokenizer path/to/src.model --tgt-tokenizer path/to/tgt.model --data path/to/preprocessed_dir --restore-file checkpoint_best.pt
```

注意：`train.py` 使用 `--cuda` 开启 GPU（代码中仍会使用 `args.cuda` 传播到模型与 batch 生成）。

## 可修改点与扩展点（Agent 可直接修改）
- 新模型：在 `seq2seq/models/` 下添加模块并用 `@register_model('name')` 与 `@register_model_architecture('name','arch')` 注册，确保实现 `build_model` 和 `add_args`。
- 解码策略：当前为贪心解码，若添加 beam，请复用 `BeamSearch` 或替换 `seq2seq/decode.py` 中的主逻辑。
- 预训练 embedding：`transformer.py` 支持通过 `--encoder-embed-path` / `--decoder-embed-path` 加载 embeddings（`utils.load_embedding` 将 token 文本映射到 id，并返回 dict）。注意：当前 transformer 中的 pretrained embedding 变量标记为“目前未使用”。

## 调试建议（具体且可执行）
- 如果训练时损失为 NaN：在 `train.py` 已有检查分支，会打印该 batch 原始句子。可在那处临时添加 `pdb` 或更细粒度的数值检查。
- 如果解码生成长度异常：确认 `transformer.pos_embed` 的 `max_seq_len` 与 `args.max_seq_len` 一致；`decode()` 中会裁剪 `generated` 到 `model.decoder.pos_embed.size(1)`。
- batch/掩码问题：`src_pad_mask` 在 `make_batch_input` 中为形状 (B,1,1,L)，解码器期望类似的广播形状（检查 `TransformerDecoder.future_mask` 的返回形状）。

## 重要文件索引（供快速跳转）
- 入口：`train.py`, `translate.py`, `preprocess.py`
- 模型：`seq2seq/models/model.py`, `seq2seq/models/transformer.py`
- 解码/搜索：`seq2seq/decode.py`, `seq2seq/beam.py`
- 数据/批处理：`seq2seq/data/dataset.py`
- 工具：`seq2seq/utils.py`

如果上述任何路径或流程需要更详细的运行例子（例如 `preprocess.py` 的 tokenizer 训练参数或生成 pickle 的精确命令），请告诉我我会把这些补充进来。
