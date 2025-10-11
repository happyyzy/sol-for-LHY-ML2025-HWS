# Goal
本次作业的目标是：写一个code agent，完成一个表格数据回归的机器学习任务，可以在[kaggle leaderboard](https://www.kaggle.com/competitions/ml-2025-spring-hw-2/leaderboard)上提交查看测试分数，可以随便选模型改pipeline。
| Baseline | Public Score |
|-----------|---------------|
| Boss      | 0.84733       |
| Medium    | 0.91179       |
| Simple    | 1.31311       |

本次作业没达到boss baseline，所以懒得贴代码。。

# My approch
简而言之试了各种办法，甚至用qwen3-480b-coder当基座，最低也只到0.861左右，不知道[kaggle leaderboard](https://www.kaggle.com/competitions/ml-2025-spring-hw-2/leaderboard)上那些刷到0.5的神仙是怎么搞的。。
另外我自己费好大力气做这个任务也就到0.80194：真-我不如人机。我自己的代码[在这](my_code-public-0-79.ipynb)。

我还发现这个给agent的任务其实是李宏毅老师[ml2023](https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php)的hw1:regression，数据集格式一模一样，只是取得大数据集的不同片段，当时这份作业的boss baseline是0.81456，
而且当时还不流行agent，这个作业是布置给学生做的，当年的[leaderboard](https://www.kaggle.com/competitions/ml2023spring-hw1/leaderboard)和现在这个
agent的[leaderboard](https://www.kaggle.com/competitions/ml-2025-spring-hw-2/leaderboard)分布特别像，都是两三个0.5开头的，之后断层到0.8+,还挺有趣的。
