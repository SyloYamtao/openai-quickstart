### 2024年03月24日

#### 实现 OpenAI-Translator V2.0 中的一个或多个特性

| 项目                      | 进度   |
|-------------------------|------|
| 支持图形用户界面（GUI）,提升易用性。 | 部分完成 |
| 添加对保留源 PDF 的原始布局的支持     | 无    |
| 服务化：以 API 形式提供翻译服务支持    | 无    |
| 添加对其他语言的支持              | 完成   |

##### GUI

相关内容存在于ai_translator/gui下

##### 添加对其他语言的支持

启动项目请在`openai-quickstart/openai-translator`路径下,
使用命令`python --model_type OpenAIModel --openai_api_key ${YOUR_OPENAI_API_KEY} --target_language JA`启动,
其中`--traget_language`参数用来指定翻译结果的语言
其中:

| 输入值 | 代表语言 |
|------|------|
| 无 | 中文 |
| JA | 日文 |
| KO | 韩文 |
| EN | 英文 |
| GE | 德语 |
| FR | 法语 |
| IT | 意大利语 |