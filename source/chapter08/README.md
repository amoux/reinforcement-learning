# Stocks Trading Using RL  📊

- Implement our own *OpenAI Gym* environments to simulate the stock market.
- Apply the ***DQN*** method from `chapter06` to train an agent to trade stocks to maximize profit.

---

- Project Structure
  - `data/` :
    - unpack_data.sh
    - ch08-small-quotez.tgz
  - `lib/`  :
    - common.py
    - models.py
    - validation.py
    - utils.py

## Trading

Types of financial instruments traded on markets: `goods, stocks, and currencies`; All these items have a price that changes over time.

> What is trading? Trading is the activity of buying and selling financial instruments with different goals.

- Trading goals
  - investment (profit)
  - hedging/futures (gaining protection from future price movements)

## Problem ( 🤔 )

The question is: can we look at the problem from the RL angle? Let's say that we have some observation of the market, and we want to make a decision:

- `{buy, sell, or wait}`
  - **decision ? reward : profit**
  - ***If*** we buy before the price *goes up*
    - ***then*** our profit will be `positive (+)` 📉
  - ***otherwise***
    - we will get a `negative` reward `(-)` 📈
  
- What we're trying to do is get as much profit as possible. The connections between market trading and **RL** are quite obvious.

### Goal

- We will investigate whether it will be possible for our agent to learn when the best time is to buy one single share and then close the position to maximize the profit.

### Data

```scala
<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>
20160104,100100,1148.9,1148.9,1148.9,1148.9,0
20160104,100200,1148.9,1148.9,1148.9,1148.9,50
20160104,100300,1149.0,1149.0,1149.0,1149.0,33
20160104,100400,1149.0,1149.0,1149.0,1149.0,4
20160104,100500,1153.0,1153.0,1153.0,1153.0,0
20160104,100600,1156.9,1157.9,1153.0,1153.0,43
20160104,100700,1150.6,1150.6,1150.4,1150.4,5
20160104,100800,1150.2,1150.2,1150.2,1150.2,4
...
```

### Key Decisions

- Observations
  - *N* past bars, where each has open, high, low and close prices.
  - An indication that the share was bought some *time-ago* (only one share at a time will be possible).
  - Profit or loss that we currently have from our current position (the share bought).

- Actions
  - **Do nothing:** *skip the bar without taking an action.*
  - **Buy a share:** *if the agent has already got the share, nothing will be bought;*
    - otherwise, we will pay the commission, which is usually some small percentage of the current price.
  - **Close the position:** *if we do not have a previously purchased share, nothing will happen;*
    - otherwise, we will pay commission for the trade.

- Rewards
  - **?A :** Split the reward into multiple steps during ownership of the share (*reward on every step will be equal to the last bar's movement*).
  - **?B :** On the other hand, the agent will receive the reward only after the close action and receive the full reward at once.

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
