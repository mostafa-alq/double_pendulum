# Double Pendulum

A double pendulum on a cart, simulated from scratch, with a neural network that learns to swing it up from hanging still and keep it balanced upside down.

![Swing-up](outputs/swingup/progress/iter_10225.gif)

![Game](docs/game.png)

Run `python game.py` to play. Push the cart with the arrow keys, or press S to let the AI swing it up and A to let it balance. H drops it back to the bottom and R stands it up.

## AI usage

Like most developers nowadays, I used Claude to help with this project, but I plan on being honest about it as I did not want to delegate my thinking to a chatbot. I wrote the original double pendulum in `fixed_simulation.py` and the physics for the cart version in `physics.py` myself, setting up the Lagrangian and deriving the equations of motion with sympy. I also learnt the fundamentals of neural networks and wrote the neural network in `neural.py` by hand so I can actually understand how it works, and I made the base of the game in pygame. Claude wrote the RK4 step, a lot of the game UI on top of what I made and the training environment in `train.py` and `cart_env.py`, though I was directing the training and decided what to try next and which version to keep.

## How it works

I worked out the equations of motion with Lagrangian mechanics, using sympy to do the algebra, and the simulation steps them forward with RK4.

Training it to be able to balance didn't take long, but the swing-up took a lot of attempts. What finally worked was training it backwards: start it just past the top so it only has to catch it, then gradually start it further and further down until it's starting from hanging still. It gets up about 98% of the time now, and usually takes around 6 seconds.

![Learning curve](outputs/swingup/progress/learning_curve.png)

The GIFs in `outputs/swingup/progress` show it at different points in training, from flailing about to swinging up properly.

## Helpful resources I used

https://matthias-research.github.io/pages/publications/XPBD.pdf

https://lpsa.swarthmore.edu/NumInt/NumIntFourth.html

https://medium.com/data-science/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
