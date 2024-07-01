# 1. 快速入门

## 1.1 argparse简介

argparse 模块可以让人轻松编写用户友好的命令行接口。程序定义它需要哪些参数，argparse 将会知道如何从 sys.argv 解析它们。argparse 模块还能自动生成帮助和用法消息文本。该模块还会在用户向程序传入无效参数时发出错误消息。

## 1.2 核心功能

### 1.2.1 创建解析器对象

argparse 模块对命令行接口的支持是围绕 argparse.ArgumentParser 的实例建立的。它是一个<font color='red'><b>用于参数规格说明的容器</b></font>并包含多个全面应用解析器的选项:

```python
import argparse

# 创建一个解析器对象
parser = argparse.ArgumentParser(
    prog='ProgramName',  # 指定程序名称为 'ProgramName'
    description='What the program does',  # 程序的描述
    epilog='Text at the bottom of help'  # 帮助信息底部的文本
)

# 让解析器对象解析参数
args = parser.parse_args()
```

🤔 𝑸𝒖𝒆𝒔𝒕𝒊𝒐𝒏：`argparse.ArgumentParser`中的三个参数有什么作用？
🥳 𝑨𝒏𝒔𝒘𝒆𝒓：在 `argparse.ArgumentParser` 构造函数中，`prog`、`description` 和 `epilog` 是三个可选参数，它们的作用如下：

1. **prog**:
   - 这个参数指定了程序的名称。它将被显示在自动生成的帮助信息和使用信息中，通常是程序的运行脚本名称或命令行调用的名称。
   - 例如，如果我们的 Python 脚本名为 `myscript.py`，我们可以将其设置为 `prog='myscript'`。

2. **description**:
   - 这个参数提供了程序的简短描述，它将在自动生成的帮助信息的开头部分显示。这个描述应该简洁明了地说明程序的主要功能或用途。
   - 例如，`description='This program processes input files and generates a report.'`。

3. **epilog**:
   - 这个参数允许我们添加一段在自动生成的帮助信息的底部显示的文本。这可以是额外的说明、注意事项或任何其他信息，它将在所有自动生成的参数说明之后显示。
   - 例如，`epilog='Please report bugs to support@example.com.'`。

这些参数都是可选的，但它们提供了一种方式来增强自动生成的帮助信息的可读性和有用性。如果这些参数没有被明确设置，`argparse` 将使用默认值或不显示某些部分。例如，如果不设置 `prog`，`argparse` 将使用实际运行脚本的名称。如果不设置 `description` 和 `epilog`，相应的部分将不会被显示。

那我们运行一下这段代码：

```bash
python a.py
```

很正常，这段代码没有输出。这是因为这段代码本身只是初始化了一个参数解析器，设置了一些基本的描述信息，这些信息将用于生成帮助文本。虽然有实际的参数解析，但我们没有给解析器对象添加任何参数，所以它没有什么作用。那我们设置的`prog`、`description`、`epilog`参数呢？这需要我们在调用该脚本的时候添加`-h`或`--help`参数：

```bash
python a.py -h

# 或者是
python a.py --help
```

结果如下：

```
usage: ProgramName [-h]

What the program does

options:
  -h, --help  show this help message and exit

Text at the bottom of help

```

我们对这个结果进行分析：

```python
usage: ProgramName [-h]  # 这一行显示了程序的用法。这里 ProgramName 应该是 prog 参数指定的程序名称。如果用户没有提供任何参数，程序将显示这个基本的用法信息。

What the program does  # 这一行是程序的简短描述（这个脚本的作用、功能是什么），对应于 ArgumentParser 构造函数中的 description 参数。

options:  # 这个标题下面是程序接受的命令行选项列表。在这个例子中，只显示了帮助选项 -h, --help（💡 因为我们没有添加其他参数，所以它只有一个默认的--help）。
  -h, --help  show this help message and exit

Text at the bottom of help  # 这一行是 epilog 参数的内容，它显示在帮助信息的最后。这可以包含额外的说明或信息，比如联系信息、版权声明等。

```

> 💡 需要注意的是，`prog`、`description`、`epilog`参数都是可选的，而且`prog`参数也不会检查当前脚本的名称是否一致（不会报错）。

### 1.2.2 为解析器添加参数

`ArgumentParser.add_argument()` 方法将单个参数规格说明关联到解析器。它支持位置参数，接受各种值的选项，以及各种启用/禁用旗标（Flag）:

```python
import argparse
from prettytable import PrettyTable
import re


def get_help_for_argument(parser, arg_name):
    for action in parser._actions:
        if action.dest == arg_name or (action.option_strings and any(opt == '--'+arg_name for opt in action.option_strings)):
            return action.help
    return "N/A"


def print_args(args, parser):
    ptable = PrettyTable(['Argument', 'Type', 'Value', 'Help/Description'])
    ptable.align = 'l'
    

    for name, value in args._get_kwargs():
        ptable.add_row(
            [
                name, 
                re.findall(pattern=r"<class '(.*)'>", string=str(type(value)))[0], 
                value, 
                get_help_for_argument(parser=parser, arg_name=name)
            ]
        )
    
    print(ptable)


def get_opts():
    # 创建一个解析器对象
    parser = argparse.ArgumentParser(
        prog='ProgramName',  # 指定程序名称为 'ProgramName'
        description='What the program does',  # 程序的描述
        epilog='Text at the bottom of help'  # 帮助信息底部的文本
    )

    # 为解析器对象添加参数
    parser.add_argument('config', help='The path of config.')  # 位置参数
    parser.add_argument('--dataset', type=str, default='Datasets/coco128', help='The dir path of dataset.') 
    parser.add_argument('--weights-path', type=str, default='runs/detect/exp/weights/best.pt', help='The path of model weights.') 
    parser.add_argument('-e', '--epoch', type=int, default=150, help='The epoch of training.')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='The learning rate of optimizer.')
    parser.add_argument('-c', '--count')  # 
    parser.add_argument('-v', '--verbose', action='store_true')  # on/off的一个flag

    # 让解析器对象解析参数
    args = parser.parse_args()
    
    return args, parser


if __name__ == '__main__':
    args, parser = get_opts()
    print_args(args, parser)
```

运行代码：

```bash
python a.py 123
```

结果如下：

```
+---------------+----------+---------------------------------+---------------------------------+
| Argument      | Type     | Value                           | Help/Description                |
+---------------+----------+---------------------------------+---------------------------------+
| config        | str      | 123                             | The path of config.             |
| dataset       | str      | Datasets/coco128                | The dir path of dataset.        |
| weights_path  | str      | runs/detect/exp/weights/best.pt | The path of model weights.      |
| epoch         | int      | 150                             | The epoch of training.          |
| learning_rate | float    | 0.0001                          | The learning rate of optimizer. |
| count         | NoneType | None                            | None                            |
| verbose       | bool     | False                           | None                            |
+---------------+----------+---------------------------------+---------------------------------+
```

这段代码中我首先定义了一个`get_help_for_argument`的函数，它是为了获取每个参数的help信息的，其次我定义了一个`print_args()`函数，这个函数调用了`PrettyTable`以及正则表达式，从而可以美化输出解析器的打印形式。

当然，我们的重点当然不是这两个函数，重点是如何给解析器对象添加参数，下面是添加参数的详细分析：

- `parser.add_argument('config', help='The path of config.')` 添加了一个位置参数 `config`，用户必须提供这个参数，它用于指定配置文件的路径。
- `parser.add_argument('--dataset', type=str, default='Datasets/coco128', help='The dir path of dataset.')` 添加了一个可选参数 `--dataset`，它有一个默认值 `'Datasets/coco128'`，用于指定数据集的目录路径。
- `parser.add_argument('--weights-path', type=str, default='runs/detect/exp/weights/best.pt', help='The path of model weights.')` 添加了 `--weights-path` 参数，用于指定模型权重文件的路径，也有一个默认值。
- `parser.add_argument('-e', '--epoch', type=int, default=150, help='The epoch of training.')` 添加了 `-e` 或 `--epoch` 参数，用于指定训练的轮数，类型为整数，有默认值。
- `parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='The learning rate of optimizer.')` 添加了 `-lr` 或 `--learning_rate` 参数，用于指定优化器的学习率，类型为浮点数，有默认值。
- `parser.add_argument('-c', '--count')` 添加了 `-c` 或 `--count` 参数，但没有指定类型、默认值或帮助信息。这可能是一个错误或遗漏，因为通常参数应该有 `help` 描述。
- `parser.add_argument('-v', '--verbose', action='store_true')` 添加了 `-v` 或 `--verbose` 参数，这是一个开关参数，当用户在命令行中提供这个参数时，`args.verbose` 将被设置为 `True`。

🤔 𝑸𝒖𝒆𝒔𝒕𝒊𝒐𝒏：什么是位置参数，什么是可选参数？
🥳 𝑨𝒏𝒔𝒘𝒆𝒓：在 `argparse` 模块中，参数主要分为两种类型：①<font color='red'><b>位置参数（Positional Arguments）</b></font>和②<font color='green'><b>可选参数（Optional Arguments）</b></font>。

1. **位置参数**：
   - 位置参数是用户在命令行中<font color='red'><b>必须提供的参数</b></font>，它们没有前缀（如 `-h` 或 `--`），并且通常按照在 `ArgumentParser` 中添加它们的顺序来接收值。
   - 位置参数通常用于程序的主要输入，例如文件名、输入数据等。
   - 在 `add_argument` 方法中添加位置参数时，只需要指定参数的名称，例如：
     ```python
     parser.add_argument('config', help='The path of config.')  # 配置文件的路径，这个是必须提供的
     ```
   - 用户在调用程序时必须按照顺序提供这些参数的值。

2. **可选参数**：
   - 可选参数是用户可以在命令行中指定，也可以不指定的参数。它们使用 `-h` 或 `-` 作为前缀，并且可以有长格式和短格式两种形式。
   - 可选参数通常用于配置程序的行为，例如设置日志级别、启用或禁用某个特性等。
   - 在 `add_argument` 方法中添加可选参数时，需要使用 `--` 来指定长格式名称，并且可以添加一个短格式名称（使用 `-`），例如：
     ```python
     parser.add_argument('--verbose', help='Enable verbose mode')
     parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0')
     ```
   - 如果用户在命令行中提供了可选参数，`argparse` 将根据参数的定义来设置相应的值或行为。

**两种参数的主要区别在于：**

- **必需性**：⚠️ <font color='red'><b>位置参数是必需的</b></font>，而可选参数可以不提供。
- **前缀**：位置参数没有前缀，可选参数使用 `-h` 或 `-` 作为前缀。
- **顺序**：位置参数在命令行中<font color='red'><b>出现的顺序很重要</b></font>，而可选参数则不需要考虑顺序。

> `argparse` 还提供了一些其他类型的参数，例如开关参数（通过 `action='store_true'` 或 `action='store_false'` 定义），它们在用户指定时设置为 `True` 或 `False`。但基本上，所有参数都可以归类为位置参数或可选参数。

### 1.2.3 解析函数

`ArgumentParser.parse_args()` 方法运行解析器并将提取的数据放入 `argparse.Namespace` 对象:

```python
# 让解析器对象解析参数
args = parser.parse_args()
```

🤔 𝑸𝒖𝒆𝒔𝒕𝒊𝒐𝒏：`argparse.Namespace` 对象是什么？
🥳 𝑨𝒏𝒔𝒘𝒆𝒓：`argparse.Namespace` 对象是 Python `argparse` 模块中用于存储解析后的命令行参数值的一种容器。当 `ArgumentParser` 的 `parse_args()` 方法被调用时，它会将用户通过命令行提供的参数值按照定义的参数属性存储在这个命名空间对象中。

以下是 `argparse.Namespace` 对象的一些关键特性：

1. **属性访问**：每个命令行参数都对应 `Namespace` 对象的一个属性。属性的名称对应参数的 `dest` 属性（默认为参数名称的长格式名称去除前缀和破折号）。例如，如果一个参数使用 `--verbose` 定义，那么在 `Namespace` 对象中可以通过 `verbose` 属性访问它的值。

2. **自动类型转换**：`Namespace` 对象根据参数定义中的 `type` 属性自动将命令行输入转换为相应的类型。例如，如果定义了一个参数类型为 `int`，那么输入的字符串将被转换为整数。

3. **默认值**：如果参数没有在命令行中提供，`Namespace` 对象将使用参数定义中指定的 `default` 值。

4. **开关参数**：对于使用 `action='store_true'` 或 `action='store_false'` 定义的参数，`Namespace` 对象将根据是否提供了该参数来设置布尔值 `True` 或 `False`。

5. **可迭代**：`Namespace` 对象是可迭代的，可以迭代出所有的属性名称和值。

6. **作为字典使用**：`Namespace` 对象的行为类似于一个只读的字典，可以使用 `args.attr` 来访问属性，其中 `attr` 是参数的 `dest` 名称。

7. **字符串表示**：将 `Namespace` 对象转换为字符串时，它将显示所有参数的名称和值，这在调试时非常有用。

# 2. 添加参数add_argument()

上面在使用 `add_argument()` 函数的时候没有对其中的细节进行说明，因此这里我们详细说明一下。

```python
ArgumentParser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
```

定义单个的命令行参数应当如何解析。每个形参都在下面有它自己更多的描述，长话短说有：

- `name` or `flags`：一个命名或者一个选项字符串的列表，例如 `foo` 或 `-f`, `--foo`。
- `action`：当参数在命令行中出现时使用的动作基本类型。
- `nargs`：命令行参数应当消耗的数目。
- `const`：被一些 `action` 和 `nargs` 选择所需求的常数。
- `default`：当参数未在命令行中出现并且也不存在于命名空间对象时所产生的值。
- `type`：命令行参数应当被转换成的类型。
- `choices`：由允许作为参数的值组成的序列。
- `required`：此命令行选项是否可省略 （仅选项可用）。
- `help`：一个此选项作用的简单描述。
- `metavar`：在使用方法消息中使用的参数值示例。
- `dest`：被添加到 `parse_args()` 所返回对象上的属性名。

## 2.1 name or flags

`add_argument()` 方法必须知道是要接收一个可选参数，如 `-f` 或 `--foo`，还是一个位置参数，如由文件名组成的列表。因此首先传递给 `add_argument()` 的参数必须是一组旗标，或一个简单的参数名称。

### 2.1.1 可选参数

```python
parser.add_argument('-e', '--epoch')  # 🌟 推荐
parser.add_argument('--epoch')  # 🌟 推荐
parser.add_argument('--epoch', '-e')  # 🌟 推荐
parser.add_argument('-e')  # ❌ 不推荐（如果只写一个变量，建议变量名写完整）
parser.add_argument('-epoch', '--e')  # ❌ 不推荐
```

在这里，我们是推荐前三种写法，后两种写法都不推荐。

> 💡 这里是为了方便演示，因此都使用了 epoch 作为变量，上面的代码在运行过程中会报错，因为同名变量已存在。

### 2.1.2 位置参数

```python
parser.add_argument('epcoh')
parser.add_argument('batch_size')
```

需要注意的是：

1. 位置参数没有`-`或`--`，因此它们的顺序是非常重要的
2. 我们在同时使用位置参数和可选参数时，一般先写位置参数，最后再写可选参数。

下面举个例子：

```python
import argparse
from prettytable import PrettyTable
import re


def get_help_for_argument(parser, arg_name):
    for action in parser._actions:
        if action.dest == arg_name or (action.option_strings and any(opt == '--'+arg_name for opt in action.option_strings)):
            return action.help
    return "N/A"


def print_args(args, parser):
    ptable = PrettyTable(['Argument', 'Type', 'Value', 'Help/Description'])
    ptable.align = 'l'
    

    for name, value in args._get_kwargs():
        ptable.add_row(
            [
                name, 
                re.findall(pattern=r"<class '(.*)'>", string=str(type(value)))[0], 
                value, 
                get_help_for_argument(parser=parser, arg_name=name)
            ]
        )
    
    print(ptable)


def get_opts():
    # 创建一个解析器对象
    parser = argparse.ArgumentParser(description='The example for Le0v1n article')

    # 为解析器对象添加参数
    parser.add_argument('config_path', type=str, default="", help='The path of configuration file')
    parser.add_argument('--batch_size', type=int, default=16, help='The size of batch for dataloader')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='The learning rate of optimizer')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)

    # 让解析器对象解析参数
    args = parser.parse_args()
    
    return args, parser


if __name__ == '__main__':
    args, parser = get_opts()
    print_args(args, parser)
```

CLI：

```bash
python b.py config.yolov8s --batch_size 128 -lr 0.00001 -v
```

运行结果：

```
+---------------+-------+---------------------+----------------------------------+
| Argument      | Type  | Value               | Help/Description                 |
+---------------+-------+---------------------+----------------------------------+
| config_path   | str   | config/yolov8s.yaml | The path of configuration file   |
| batch_size    | int   | 128                 | The size of batch for dataloader |
| learning_rate | float | 1e-05               | The learning rate of optimizer   |
| verbose       | bool  | True                | None                             |
+---------------+-------+---------------------+----------------------------------+
```

## 2.2 action

`ArgumentParser` 对象将命令行参数与动作相关联。这些动作可以做与它们相关联的命令行参数的任何事，尽管大多数动作只是简单的向 `parse_args()` `返回的对象上添加属性。action` 命名参数指定了这个命令行参数应当如何处理。

### 2.2.1 store

在 `argparse` 模块中，`store` 是 `action` 参数的一个选项，用于指定当命令行中出现某个参数时应该采取的动作。`store` 动作的目的是将命令行中提供的值存储到 `argparse.Namespace` 对象的属性中。这是 `argparse` 中最常见的动作之一（也是默认的动作）。

以下是使用 `store` 动作时的一些关键点：

1. **存储值**：当 `action` 设置为 `store` 时（这是默认值），如果用户在命令行中提供了参数，其值将被存储在 `Namespace` 对象的相应属性中。

2. **类型转换**：`argparse` 会根据 `add_argument` 方法中指定的 `type` 参数自动将输入的字符串转换为相应的类型。例如，如果 `type=int`，则输入的字符串将被转换为整数。

3. **默认值**：如果用户没有在命令行中提供该参数，且我们提供了 `default` 值，`Namespace` 对象的相应属性将被设置为 `default` 值。

4. **必需性**：如果参数是必需的（即 `required=True`），并且用户没有在命令行中提供该参数，`argparse` 将报错并显示帮助信息。

下面是一个使用 `store` 动作的例子：

```python
parser.add_argument('--value', type=int, default=10, help='An integer value')
```

```
+----------+------+-------+------------------+
| Argument | Type | Value | Help/Description |
+----------+------+-------+------------------+
| value    | int  | 10    | An integer value |
+----------+------+-------+------------------+
```

在这个例子中，如果用户在命令行中使用 `--value` 参数并提供了一个值，例如 `--value 20`，`args.value` 将被设置为 `20`。如果用户没有提供 `--value` 参数，`args.value` 将使用默认值 `10`。

`store` 动作是处理命令行参数的标准方式，适用于大多数需要接受用户输入的场景。

### 2.2.2 store_const<a id=store_const></a>

在 `argparse` 模块中，`store_const` 是一个 `action` 参数的选项，它用于指定当命令行中出现某个参数时应该采取的动作。当 `action` 被设置为 `store_const` 时，无论命令行中是否提供了该参数的值，存储的值都将被设置为在 `add_argument` 方法中指定的常量值。

以下是使用 `store_const` 时的一些关键点：

1. **常量值**：使用 `store_const` 时，我们需要提供一个 `const` 参数，这个参数的值将被存储在 `Namespace` 对象的相应属性中。

2. **忽略命令行值**：<font color='red'><b>使用户在命令行中提供了值，这个值也会被忽略</b></font>，存储的将是 `const` 参数指定的常量值。

3. **默认行为**：如果用户没有在命令行中提供该参数，`Namespace` 对象的相应属性将不会被设置，除非我们同时提供了 `default` 参数。

下面是一个使用 `store_const` 的例子：

```python
parser.add_argument('--mode', action='store_const', const='DEBUG', default='INFO', help='Set the mode to const value')
```

运行结果：

```bash 
# 🪐 第一种情况：如果我们没有在CLI中开启这个参数，则为default值
python c.py

+----------+------+-------+-----------------------------+
| Argument | Type | Value | Help/Description            |
+----------+------+-------+-----------------------------+
| mode     | str  | INFO  | Set the mode to const value |
+----------+------+-------+-----------------------------+

# 🪐 第二种情况：如果我们在CLI中开启这个参数，则为const值
python c.py --mode
+----------+------+-------+-----------------------------+
| Argument | Type | Value | Help/Description            |
+----------+------+-------+-----------------------------+
| mode     | str  | DEBUG | Set the mode to const value |
+----------+------+-------+-----------------------------+
```

在这个例子中：
- 如果用户没有提供 `--mode` 参数，`args.mode` 将使用默认值 `'INFO'`。
- 如果用户在命令行中使用 `--mode` 参数，`args.mode` 都将被设置为 `'DEBUG'`。

**总结**：使用 `store_const` 可以方便地根据某个选项的存在与否来设置一个固定的值，而不需要关心选项的具体值是什么。<font color='green'><b>这在需要根据一个Flag来启用或禁用某些功能时非常有用</b></font>。

### 2.2.3 store_true和store_false

`argparse` 是 Python 的一个模块，用于编写用户友好的命令行接口。程序定义它需要的参数，然后 `argparse` 将自动生成帮助和使用说明，并处理与命令行参数解析相关的任务。

在 `argparse` 中，`store_true` 和 `store_false` 是两种用于处理布尔标志（也称为开关或选项）的 action 类型。

#### 2.2.3.1 store_true

- 当使用 `store_true` 作为 action 时，<font color='red'><b>如果命令行中出现了这个选项</b></font>，那么对应的变量将被设置为 `True`。
- 如果没有指定这个选项，那么变量将保持其默认值，通常是 `False`。

例子：

```python
parser.add_argument('--flag_1', '-f1', action='store_true', help='如果命令行出现了那么这个值就是True，否则就是False')
parser.add_argument('--flag_2', '-f2', action='store_true', help='如果命令行出现了那么这个值就是True，否则就是False')
```

CLI：

```bash
python Python/argparse/code/c.py -f1
```

结果如下：

```
+----------+------+-------+---------------------------------------------------+
| Argument | Type | Value | Help/Description                                  |
+----------+------+-------+---------------------------------------------------+
| flag_1   | bool | True  | 如果命令行出现了那么这个值就是True，否则就是False |
| flag_2   | bool | False | 如果命令行出现了那么这个值就是True，否则就是False |
+----------+------+-------+---------------------------------------------------+
```

可以发现，因为我们在CLI中使用了`-f1`，它被触发了，所以值为`True`，而`-f2`这个参数没有在CLI中使用，没有被触发，所以它的值还是`False`。

#### 2.2.3.2 store_false

- 类似地，`store_false` 用于设置变量为 `False`。
- 如果命令行中出现了这个选项，那么对应的变量将被设置为 `False`。
- 如果没有指定这个选项，变量将保持其默认值，通常是 `True`。

例子：

```python
parser.add_argument('--flag_1', '-f1', action='store_false', help='如果命令行出现了那么这个值就是False，否则就是True')
parser.add_argument('--flag_2', '-f2', action='store_false', help='如果命令行出现了那么这个值就是False，否则就是True')
```

CLI：

```bash
python Python/argparse/code/c.py -f1
```

结果：

```
+----------+------+-------+---------------------------------------------------+
| Argument | Type | Value | Help/Description                                  |
+----------+------+-------+---------------------------------------------------+
| flag_1   | bool | False | 如果命令行出现了那么这个值就是False，否则就是True |
| flag_2   | bool | True  | 如果命令行出现了那么这个值就是False，否则就是True |
+----------+------+-------+---------------------------------------------------+
```

可以看到，`store_false`与`store_true`的效果是反着的。

> 💡 `store_false`这种很少见，我们可以忘记它🤣。

### 2.2.4 append<a id=append></a>

在 `argparse` 模块中，`append` 是一个 action 类型，它用于处理命令行参数，<font color='red'><b>允许用户为某个参数多次传入值，并将这些值收集到一个列表中</b></font>。这在需要接受多个值作为输入时非常有用。

**例子1**：

```python
parser.add_argument('--items', action='append', help='可以多次指定的参数')
```

**CLI**：

```bash
python Python/argparse/code/c.py
```

**结果**：

```
+----------+----------+-------+--------------------+
| Argument | Type     | Value | Help/Description   |
+----------+----------+-------+--------------------+
| items    | NoneType | None  | 可以多次指定的参数 |
+----------+----------+-------+--------------------+
```

我们可以发现，如果我们在CLI中什么都不传给它，它也不会报错，但它的默认值是None。

**例子2**：

```python
parser.add_argument('--items', action='append', default=['Le0v1n'], help='可以多次指定的参数')
```

**CLI**：

```bash
python Python/argparse/code/c.py
```

**结果**：

```
+----------+------+------------+--------------------+
| Argument | Type | Value      | Help/Description   |
+----------+------+------------+--------------------+
| items    | list | ['Le0v1n'] | 可以多次指定的参数 |
+----------+------+------------+--------------------+
```

这说明这个变量如果我们没有指定默认值，那么它的默认值就是`None`，否则就是我们指定的值。

**例子3**：

```python
parser.add_argument('--items', '-i', action='append', default=['Le0v1n'], help='可以多次指定的参数')
```

**CLI**：

```bash
python Python/argparse/code/c.py --items Le0v1n --items leovin -i 123 -i "Tom" -i Jerry
```

**结果**：

```
+----------+------+-------------------------------------------------------+--------------------+
| Argument | Type | Value                                                 | Help/Description   |
+----------+------+-------------------------------------------------------+--------------------+
| items    | list | ['Le0v1n', 'Le0v1n', 'leovin', '123', 'Tom', 'Jerry'] | 可以多次指定的参数 |
+----------+------+-------------------------------------------------------+--------------------+
```

这说明：

- `action='append'`可以接收多个变量（不要有空格）
- 在CLI中，`"  "` 和 `  ` 以及 `'   '` 都是可以的
- 在CLI中，数字也会被认为是字符串

### 2.2.5 append_const

`argparse` 中的 `append_const` 是一个 action 类型，它允许我们将一个固定的值（常量）添加到列表中，而不是添加命令行参数本身的值。这在需要基于某个选项多次添加相同值到列表时非常有用。

> 要想了解append_const，我们应该先理解[store_const](#store_const)和[append](#append)。

**例子**：

```python
parser.add_argument('--trigger', action='append_const', const='It has been triggerred!', help='添加常量值到列表')
```

**CLI**：

```bash
python Python/argparse/code/c.py --trigger --trigger --trigger
```

**结果**：

```
+----------+------+-----------------------------------------------------------------------------------+------------------+
| Argument | Type | Value                                                                             | Help/Description |
+----------+------+-----------------------------------------------------------------------------------+------------------+
| trigger  | list | ['It has been triggerred!', 'It has been triggerred!', 'It has been triggerred!'] | 添加常量值到列表 |
+----------+------+-----------------------------------------------------------------------------------+------------------+
```

**注意事项**：

- `append_const` 需要与 `const` 参数一起使用，`const` 参数定义了要添加到列表中的常量值。
- 如果没有指定 `const` 参数，会报错。
- 使用 `append_const` 时，每次指定参数都会添加一个新的常量值到列表中，无论命令行中该参数出现多少次。

> 💡 `append_const`其实在深度学习领域用的非常少，我们大概了解一下就可以，不必深究。


### 2.2.6 count

在 `argparse` 模块中，`count` 是一个 action 类型，它用于计算某个选项在命令行中出现的次数，并将这个次数存储在指定的变量中。这个特性<font color='red'><b>在我们需要统计某个选项被提及的次数时非常有用</b></font>。

当我们为某个参数指定 `action='count'` 时，每次在命令行中指定这个参数，计数器就会增加 1。这个计数器的值最终会赋给对应的变量。

**例子1**：

```python
parser.add_argument('--verbose', '-v', action='count')
```

**CLI**：

```bash
python Python/argparse/code/c.py
```

**结果**：

```
+----------+----------+-------+------------------+
| Argument | Type     | Value | Help/Description |
+----------+----------+-------+------------------+
| verbose  | NoneType | None  | None             |
+----------+----------+-------+------------------+
```

**例子2**：

```python
parser.add_argument('--verbose', '-v', action='count')
```

**CLI**：

```bash
python Python/argparse/code/c.py -v -v -v -v -v -v -v
```

**结果**：

```
+----------+------+-------+------------------+
| Argument | Type | Value | Help/Description |
+----------+------+-------+------------------+
| verbose  | int  | 7     | None             |
+----------+------+-------+------------------+
```

**例子3**：

```python
parser.add_argument('--verbose', '-v', default=1, action='count')
```

**CLI**：

```bash
python Python/argparse/code/c.py -vvvvvvv
```

**结果**：

```
+----------+------+-------+------------------+
| Argument | Type | Value | Help/Description |
+----------+------+-------+------------------+
| verbose  | int  | 7     | None             |
+----------+------+-------+------------------+
```

**注意事项**：

- 默认情况下，如果没有为使用 `count` 的参数指定任何值，`args` 中对应的属性将是 `None`。如果我们希望在没有提供任何值时有一个默认的计数器值，我们可以使用 `default` 参数：
  ```python
  parser.add_argument('--verbose', '-v', action='count', default=1)
  ```
  这样，即使没有指定 `-v`，详细级别也会从 1 开始计数。

- `count` action 不接受任何额外的值，它只是简单地增加计数器。

- 通常，`count` 用于创建一个递增的详细级别，比如 `-v`、`-vv`、`-vvv` 这样的使用模式，其中每个 `-v` 都增加详细级别。

`count` 是处理需要统计出现次数的参数时非常有用的一个选项，它使得命令行接口更加灵活和强大，尤其是在需要根据用户指定的次数来调整程序行为的场景中。

> 💡 这个在深度学习领域中其实也没啥用，我们直接定义一个int类型的变量就行，没必要这样写🤣。

### 2.2.7 version

在 `argparse` 模块中，`version` 是一个特殊的 action 类型，它用于自动处理版本信息的显示。当使用 `action='version'` 时，我们可以指定一个版本字符串，当用户在命令行中使用 `--version` 或 `-v`（取决于如何定义参数）选项时，程序将自动打印出这个版本字符串并退出。

要使用 `version` action，我们通常需要定义一个参数（通常是 `--version`），并将其 `action` 设置为 `'version'`。同时，我们需要提供一个 `version` 字符串，这个字符串将被用作程序的版本信息。

**例子1**：

```python
def get_opts():
    # 创建一个解析器对象
    parser = argparse.ArgumentParser(description='The example for Le0v1n article')

    # 为解析器对象添加参数
    print(f"---------- [1] parser对象开始添加参数... ----------")
    parser.add_argument('--version', '-v', action='version', version='%(prog)s 1.0', help='显示版本信息并退出')
    parser.add_argument('--epoch', '-e', type=int, default=150, help='The epoch of training')
    print(f"---------- [2] parser对象添加参数已完成！ ----------")

    # 让解析器对象解析参数
    print(f"---------- [3] parser对象开始解析参数... ----------")
    args = parser.parse_args()
    print(f"---------- [4] parser对象解析参数已完成！ ----------")
    
    return args, parser
```

**CLI**：

```bash
python Python/argparse/code/c.py
```

**结果**：

```
---------- [1] parser对象开始添加参数... ----------
---------- [2] parser对象添加参数已完成！ ----------
---------- [3] parser对象开始解析参数... ----------
---------- [4] parser对象解析参数已完成！ ----------
+----------+------+-------+-----------------------+
| Argument | Type | Value | Help/Description      |
+----------+------+-------+-----------------------+
| epoch    | int  | 150   | The epoch of training |
+----------+------+-------+-----------------------+
```

说明程序正常解析了，且`args`中没有`--version`参数。

**例子2**：

```python
def get_opts():
    # 创建一个解析器对象
    parser = argparse.ArgumentParser(description='The example for Le0v1n article')

    # 为解析器对象添加参数
    print(f"---------- [1] parser对象开始添加参数... ----------")
    parser.add_argument('--version', '-v', action='version', version='%(prog)s 1.0', help='显示版本信息并退出')
    parser.add_argument('--epoch', '-e', type=int, default=150, help='The epoch of training')
    print(f"---------- [2] parser对象添加参数已完成！ ----------")

    # 让解析器对象解析参数
    print(f"---------- [3] parser对象开始解析参数... ----------")
    args = parser.parse_args()
    print(f"---------- [4] parser对象解析参数已完成！ ----------")
    
    return args, parser
```

**CLI**：

```bash
python Python/argparse/code/c.py -v
```

**结果**：

```
---------- [1] parser对象开始添加参数... ----------
---------- [2] parser对象添加参数已完成！ ----------
---------- [3] parser对象开始解析参数... ----------
c.py 1.0
```

我们可以看到，因为在CLI中添加了`-v`或`--version`，此时`version`变量起作用了，前面在介绍`version`时我们就强调过，查看完版本后脚本会自动结束运行，所以这里的`[4]`并没有被打印出来，即`parser`对象在解析完参数时，发现`version`被启用了，所以打印了版本号并直接停止了程序。

**注意事项**：

- `version` action 通常与 `--version` 参数一起使用，但我们可以根据需要定义其他参数名。
- 版本字符串可以使用格式化，`%(prog)s` 是一个特殊的占位符，代表脚本的名称。
- 使用 `version` action 可以避免手动编写代码来处理版本信息的显示，`argparse` 会自动处理这一切。
- 当 `--version` 被触发时，<font color='red'><b>程序会立即退出（在解析完参数后），不会执行其他任何操作</b></font>。

> `version` action 是 `argparse` 提供的一个方便的特性，它使得显示程序版本信息变得非常简单和标准化。这对于维护用户友好的命令行界面非常有帮助。
>
> 💡 其实我们可以不用`action='version'`而是直接定义一个`str`的version，但前者可以通过CLI查看版本信息，后者不可以。

### 2.2.8 extend

`action=extend`和`action='append'`是相反的，后者是追加，前者是在开头添加，这里不再过多的赘述。

## 2.3 choices

在 `argparse` 模块中，`choices` 是一个参数选项，<font color='red'><b>用于限制命令行参数的值只能是预定义的一组选项中的一个</b></font>。使用 `choices` 可以确保用户输入的参数值是有效和预期的。

当我们为参数定义时添加 `choices` 选项，我们可以提供一个列表`list`或集合`set`，其中包含所有允许的值。如果用户输入的值不在 `choices` 列表中，`argparse` 将抛出一个错误，并显示有效的选项。

**例子1**

```python
parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='sgd', help='The optimizer of training, including sgd, adam')
```

**CLI**：

```bash
python Python/argparse/code/c.py
```

结果：

```
+-----------+------+-------+------------------------------------------------+
| Argument  | Type | Value | Help/Description                               |
+-----------+------+-------+------------------------------------------------+
| optimizer | str  | sgd   | The optimizer of training, including sgd, adam |
+-----------+------+-------+------------------------------------------------+
```

> 💡 如果我们不设置`default`，那么默认是`None`。

**例子2**

```python
parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='sgd', help='The optimizer of training, including sgd, adam')
```

**CLI**：

```bash
python Python/argparse/code/c.py --optimizer SGD
```

结果：

```
usage: c.py [-h] [--optimizer {sgd,adam}]
c.py: error: argument --optimizer: invalid choice: 'SGD' (choose from 'sgd', 'adam')
```

**例子3**

```python
# 💡 请注意：如果要使用数字，一定一定要添加type=int，否则CLI默认会将数字转换为str，这样会报错的！
parser.add_argument('--log_level', choices=[0, 1, 2, 3], type=int, default=0, help='The level of log, including 0, 1, 2, 3')
```

**CLI**：

```bash
python Python/argparse/code/c.py --log_level 0
```

结果：

```
+-----------+------+-------+----------------------------------------+
| Argument  | Type | Value | Help/Description                       |
+-----------+------+-------+----------------------------------------+
| log_level | int  | 0     | The level of log, including 0, 1, 2, 3 |
+-----------+------+-------+----------------------------------------+
```

**注意事项**：

- `choices` 选项接受一个列表或集合，包含所有允许的参数值。
- 使用 `choices` 可以帮助验证用户输入，确保参数值符合预期。
- 当用户输入无效的值时，`argparse` 会自动提供一个错误消息，列出所有有效的选项。
- `choices` 可以与 `type` 选项结合使用，以进一步验证参数值的类型和有效性。
- 💡 请注意 `choices` 序列包含的内容会在执行任意 `type` <font color='red'><b>转换之后被检查</b></font>，因此 `choices` 序列中对象的类型应当与指定的 `type` 相匹配。

> 💡 `choices` 是 `argparse` 中一个非常有用的参数选项，它提高了命令行参数的健壮性和用户体验，通过限制用户输入的范围，可以避免许多常见的错误和问题。

## 2.4 metavar

在 `argparse` 模块中，`metavar` 是一个参数选项，<font color='red'><b>用于指定在生成的帮助信息和使用消息中，命令行参数应该显示的值。这通常用于为参数提供一个更人性化或更具体的占位符名称</b></font>。

当我们为参数定义时添加 `metavar` 选项，我们可以提供一个字符串，这个字符串将作为该参数的占位符显示在帮助信息中。这对于需要向用户明确显示参数预期类型或格式的情况特别有用。

**例子1**：

```python
parser.add_argument('--ckpt', metavar='CheckPoint', type=str, default='weights/yolov8s.pt', help='The path of model weights')
```

**CLI**：

```
python Python/argparse/code/c.py -h
```

结果：

```
usage: c.py [-h] [--ckpt CheckPoint]

The example for Le0v1n article

optional arguments:
  -h, --help         show this help message and exit
  --ckpt CheckPoint  The path of model weights
```

**注意事项**:

- `metavar` 并不影响命令行参数的实际解析，它只影响帮助信息的显示。
- 使用 `metavar` 可以帮助用户理解参数的预期类型或格式，例如，使用 `FILE` 或 `DIRECTORY` 作为文件或目录路径的占位符。
- `metavar` 值通常使用大写字母来区分于实际的参数名。
- 如果没有指定 `metavar`，`argparse` 将使用参数的名称作为占位符。

> `metavar` 是一个有用的选项，它提高了帮助信息的可读性和用户理解度，使得命令行接口更加友好和直观。

## 2.5 nargs

在 `argparse` 模块中，`nargs` 是一个参数选项，<font color='red'><b>用于指定命令行参数应该消耗的命令行参数数量</b></font>。这个选项让我们可以定义一个参数可以接受一个值、多个值，或者甚至不消耗任何值。

- `nargs` 的值可以是整数、`?`、`*` 或 `+`：
  - **整数**：指定参数必须恰好消耗 `nargs` 指定数量的命令行参数。
  - `?`：参数可选地消耗一个命令行参数。如果使用了该选项，还应指定 `default` 参数，用于在没有提供命令行参数时提供一个默认值。
  - `*`：参数消耗零个或多个命令行参数，并将所有值以列表形式返回。
  - `+`：参数消耗一个或多个命令行参数，并将所有值以列表形式返回。

**简单理解**：

- `?`表示0个或者1个
- `+`表示至少有一个
- `*`表示任意个

**例子**：

```python
parser.add_argument('--input', '-i', nargs='+', help='输入文件路径（至少有一个参数）')
parser.add_argument('--output', '-o', nargs=1, help='输出文件路径（只能有一个）')
parser.add_argument('--log_level', '-log', nargs='?', default='INFO', type=str, choices=['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'], help='日常级别，默认为INFO')
```

**CLI**：

```bash
python Python/argparse/code/c.py -i dataset/coco128/images/1.png dataset/coco128/images/2.png -o runs/detect/runs -log WARNING
```

**结果**：

```
+-----------+------+------------------------------------------------------------------+--------------------------------+
| Argument  | Type | Value                                                            | Help/Description               |
+-----------+------+------------------------------------------------------------------+--------------------------------+
| input     | list | ['dataset/coco128/images/1.png', 'dataset/coco128/images/2.png'] | 输入文件路径（至少有一个参数） |
| output    | list | ['runs/detect/runs']                                             | 输出文件路径（只能有一个）     |
| log_level | str  | WARNING                                                          | 日常级别，默认为INFO           |
+-----------+------+------------------------------------------------------------------+--------------------------------+
```

**注意事项**：

- `nargs` 允许我们灵活地定义参数如何从命令行接收值。
- 使用 `nargs='*'` 或 `nargs='+'` 时，如果用户没有为参数提供任何值，`args` 中对应的属性将是空列表 `[]`。
- 当 `nargs='?'` 且没有提供命令行参数时，将使用 `default` 指定的值。
- `nargs` 可以与 `action` 结合使用，例如 `action='store'` 或 `action='append'`，以定义参数如何存储值。

`nargs` 是 `argparse` 中一个非常强大的特性，它提供了对命令行参数解析的精细控制，使得我们可以创建更复杂和灵活的命令行接口。

## 2.6 required

加上之后该参数必须在CLI中被体现，否则就会报错。

## 2.7 type

在 `argparse` 模块中，`type` 是一个参数选项，用于指定命令行参数应该被转换成的类型。这允许我们验证用户输入的参数值，并将其转换成适当的 Python 数据类型，比如 `int`、`float`、`str` 等。

当我们为参数定义时添加 `type` 选项，我们可以提供一个 Python 类型或者一个函数，用于将输入的字符串转换成期望的类型。如果转换失败，`argparse` 将抛出一个错误。

`type`在前面我们已经用了很多了，这里不再赘述，这里唯一要强调的一点是：<font color='red'><b>不建议将 bool() 函数用作类型转换器</b></font>。这是因为它所做的只是将空字符串转为 `False` 而将非空字符串转为 `True`。这通常不是用户所想要的。

通常，`type` 关键字是仅应被用于只会引发上述三种被支持的异常的简单转换的便捷选项。任何具有更复杂错误处理或资源管理的转换都应当在参数被解析后由下游代码来完成。

### 注意事项
- `type` 可以是任何 Python 内置类型，比如 `int`、`float`、`str`、`list` 等。
- 我们也可以定义自己的类型转换函数，该函数接受一个字符串参数，并返回转换后的值。
- 使用 `type` 可以帮助验证用户输入，确保参数值符合预期的数据类型。
- 如果用户输入的值无法被转换成指定类型，`argparse` 将自动提供一个错误消息，指出转换失败。
- 一般不会使用`type=bool`，因为可能会有问题。

# 参考/知识来源

1. [Python官方文档：argparse --- 用于命令行选项、参数和子命令的解析器](https://docs.python.org/zh-cn/3/library/argparse.html)