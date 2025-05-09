** This is my notes while reading the book `make your own nueral network`**
** it might come off as nonsense to you**

computers are stupid machines that can do a lot of calculation.

the linear relation between input and output lets us make process.
this linear relationship means we can expect same result.

a simple visual

input ---> calculation on input ---> output

when we do a process we can compare the truth with the calculated answer,
this will give us the `error` .

by changing cal (i.e calculation) based on our error we get closer to truth.

when our last error is smaller than current error we nudge our c,
a litile less. we are getting close.

*quote:* That’s intuitively right - a big error
means a bigger correction is needed, and a tiny error means we need the teeniest
of nudges to c.

this principle of pridiction can be applied to clasifinig stuf as well.
it is all just a linear equation.

truth example used to train the models called training data.

how do we train a clasifier?

again we will have an equation and get an error based on our data then we change the variable to get closer to answer.

y = Ax

ladybird: length 1.0, width 3.0

after ploting in width as x inside the equation we get y which should be close to length.
       A       x
y = (0.25) * (3.0) = 0.75

we have our error : 1 - 0.75 = 0.25

> also we want to go above the 1.0 its a classifire

error = (desired target - actual output)
E = 1.1 - 0.75 = 0.35

there is some math and prof in chapter `training a simple classifire`

The moderating factor is often called a learning rate, and we’ve called it L.

A simple linear classifier is not useful
if the underlying problem is not separable by a straight line.
