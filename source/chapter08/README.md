# Stocks Trading Using RL  📊

- Implement our own *OpenAI Gym* environments to simulate the stock market.
- Apply the ***DQN*** method from `chapter06` to train an agent to trade stocks to maximize profit.

## Trading

Types of financial instruments traded on markets: `goods, stocks, and currencies`; All these items have a price that changes over time.

> What is trading? Trading is the activity of buying and selling financial instruments with different goals.

- Trading goals
  - investment (profit)
  - hedging/futures (gaining protection from future price movements)

## Problem ( 🤔 )

The question is: can we look at the problem from the RL angle?

Let's say that we have some observation of the market, and we want to make a decision:

- decisions<📊> `{buy, sell, or wait}`
  
  - **condition ? reward : profit**

  - ***If*** we buy before the price *goes up+*;
    - ***then;*** our profit will be `positive (+)` 📉

  - ***otherwise;***
    - we will get a `negative` reward `(-)` 📈
  
What we're trying to do is get as much profit as possible. The connections between market trading and **RL** are quite obvious.

- chapter08 source

  - `./data/` :
    - unpack_data.sh
    - ch08-small-quotez.tgz

  - `./lib/`  :
    - common.py
    - models.py
    - validation.py
    - utils.py

---

As a quick reference the following is DQN model used in `chapter06`.

```python
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
```
