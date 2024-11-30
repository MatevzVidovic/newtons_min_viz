from manimlib import *

class SquareToCircle(Scene):
    def construct(self):
        circle = Circle()
        circle.set_fill(BLUE, opacity=0.5)
        circle.set_stroke(BLUE_E, width=4)
        square = Square()

        self.play(ShowCreation(square))
        self.wait()
        self.play(ReplacementTransform(square, circle))
        self.wait()



        # Stretched 4 times in the vertical direction
        play(circle.animate.stretch(4, dim=0))
        # Rotate the ellipse 90°
        play(Rotate(circle, TAU / 4))
        # Move 2 units to the right and shrink to 1/4 of the original
        play(circle.animate.shift(2 * RIGHT), circle.animate.scale(0.25))
        # Insert 10 curves into circle for non-linear transformation (no animation will play)
        circle.insert_n_curves(10)
        # Apply a complex transformation of f(z)=z^2 to all points on the circle
        play(circle.animate.apply_complex_function(lambda z: z**2))
        # Close the window and exit the program
        exit()