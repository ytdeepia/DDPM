from manim import *
import sys

sys.path.append("../")


class Scene2_2(Scene):
    def construct(self):

        # Describe the noising and notations
        self.next_section(skip_animations=False)

        circle_x0 = Circle(radius=0.5, color=WHITE)
        x0 = MathTex("x_0", color=WHITE, font_size=32).move_to(circle_x0.get_center())

        ffhq = (
            ImageMobject("img/ffhq_1.png")
            .scale_to_fit_width(2)
            .next_to(circle_x0, UP, buff=1.0)
        )
        rect_ffhq = SurroundingRectangle(ffhq, color=WHITE, buff=0.0)

        line_x0 = DashedLine(circle_x0.get_top(), rect_ffhq.get_bottom(), color=WHITE)

        self.play(LaggedStart(Create(circle_x0), Write(x0), lag_ratio=0.5))
        self.play(
            LaggedStart(Create(line_x0), FadeIn(ffhq, rect_ffhq), lag_ratio=0.8),
            run_time=1.5,
        )

        self.play(
            Group(circle_x0, x0, line_x0, ffhq, rect_ffhq).animate.shift(LEFT * 2)
        )

        circle_x1 = Circle(radius=0.5, color=WHITE).shift(RIGHT * 2)
        x1 = MathTex("x_1", color=WHITE, font_size=32).move_to(circle_x1.get_center())

        ffhq_noise1 = (
            ImageMobject("img/ffhq_1_noise_30.png")
            .scale_to_fit_width(2)
            .next_to(circle_x1, UP, buff=1.0)
        )
        rect_ffhq_noise1 = SurroundingRectangle(ffhq_noise1, color=WHITE, buff=0.0)

        line_x1 = DashedLine(
            circle_x1.get_top(), rect_ffhq_noise1.get_bottom(), color=WHITE
        )

        arrow_q1 = Arrow(circle_x0.get_right(), circle_x1.get_left(), color=WHITE)
        label_q1 = MathTex("q(x_1|x_0)", color=WHITE, font_size=32).next_to(
            arrow_q1, UP
        )

        self.play(
            LaggedStart(
                GrowArrow(arrow_q1),
                AnimationGroup(Create(circle_x1), Write(x1)),
                lag_ratio=0.5,
            )
        )
        self.play(
            LaggedStart(
                Create(line_x1),
                FadeIn(ffhq_noise1, rect_ffhq_noise1),
                lag_ratio=0.8,
            ),
            run_time=1.5,
        )
        self.play(Write(label_q1))

        eq_1 = MathTex(
            r"x_1 = x_0 + \beta \cdot \epsilon", color=WHITE, font_size=32
        ).shift(1 * DOWN)
        eq_2 = MathTex(
            r"\epsilon \sim \mathcal{N}(0, 1)", color=WHITE, font_size=32
        ).next_to(eq_1, DOWN, buff=0.5)

        self.play(Write(eq_1))
        self.play(Write(eq_2))

        eq_3 = MathTex(
            r"q(x_1|x_0) = \mathcal{N}(x_0, \beta)", color=WHITE, font_size=32
        ).next_to(arrow_q1, DOWN, buff=0.5)
        self.play(FadeOut(eq_1, eq_2))

        self.play(Transform(label_q1, eq_3))

        # Iterate over to the next noise level
        self.next_section(skip_animations=False)

        step1 = Group(
            circle_x0,
            x0,
            line_x0,
            ffhq,
            rect_ffhq,
            circle_x1,
            x1,
            line_x1,
            ffhq_noise1,
            rect_ffhq_noise1,
            arrow_q1,
            label_q1,
        )

        self.play(step1.animate.shift(LEFT * 2), run_time=1.5)

        circle_x2 = Circle(radius=0.5, color=WHITE).next_to(circle_x1, RIGHT, buff=3.0)
        x2 = MathTex("x_2", color=WHITE, font_size=32).move_to(circle_x2.get_center())

        ffhq_noise2 = (
            ImageMobject("img/ffhq_1_noise_61.png")
            .scale_to_fit_width(2)
            .next_to(circle_x2, UP, buff=1.0)
        )
        rect_ffhq_noise2 = SurroundingRectangle(
            ffhq_noise2, color=WHITE, buff=0.0, z_index=2
        )

        line_x2 = DashedLine(
            circle_x2.get_top(), rect_ffhq_noise2.get_bottom(), color=WHITE
        )

        arrow_q2 = Arrow(circle_x1.get_right(), circle_x2.get_left(), color=WHITE)
        label_q2 = MathTex("q(x_2 \mid x_1)", color=WHITE, font_size=32).next_to(
            arrow_q2, DOWN, buff=0.5
        )

        self.play(
            LaggedStart(
                GrowArrow(arrow_q2),
                AnimationGroup(Create(circle_x2), Write(x2)),
                lag_ratio=0.5,
            )
        )
        self.play(
            LaggedStart(
                Create(line_x2), FadeIn(ffhq_noise2, rect_ffhq_noise2), lag_ratio=0.8
            ),
            run_time=1.5,
        )

        self.play(Write(label_q2))

        eq_4 = MathTex(
            r"x_2 = x_1 + \beta \cdot \epsilon", color=WHITE, font_size=32
        ).next_to(arrow_q2, UP, buff=0.5)

        self.play(Write(eq_4))

        eq_6 = MathTex(
            r"q(x_2|x_1) = \mathcal{N}(x_1, \beta)", color=WHITE, font_size=32
        ).next_to(arrow_q2, DOWN, buff=0.5)
        self.play(Transform(label_q2, eq_6))

        circle_x0_cp = VGroup(circle_x0, x0).copy()
        circle_x2_cp = VGroup(circle_x2, x2).copy()

        self.play(VGroup(circle_x0_cp, circle_x2_cp).animate.shift(3 * DOWN))

        arrow_q0_q2 = Arrow(
            circle_x0_cp.get_right(), circle_x2_cp.get_left(), color=WHITE
        )
        eq_7 = MathTex(
            r"x_2 = x_0 + 2 \beta \cdot \epsilon", color=WHITE, font_size=32
        ).next_to(arrow_q0_q2, UP, buff=0.5)

        label_q0_q2 = MathTex(
            r"q(x_2|x_0) = \mathcal{N}(x_0, 2 \beta)", color=WHITE, font_size=32
        ).next_to(arrow_q0_q2, UP, buff=0.5)

        self.play(GrowArrow(arrow_q0_q2))

        self.play(Write(eq_7))
        self.play(ReplacementTransform(eq_7, label_q0_q2))

        # Iterate to time t
        self.next_section(skip_animations=False)

        xt = MathTex("x_t", color=WHITE, font_size=32).move_to(circle_x2.get_center())
        noise = (
            ImageMobject("img/pure_noise_198.png", z_index=-1)
            .scale_to_fit_width(2)
            .move_to(ffhq_noise2.get_center())
        )

        self.play(
            FadeOut(
                circle_x1,
                x1,
                line_x1,
                ffhq_noise1,
                rect_ffhq_noise1,
                arrow_q1,
                arrow_q2,
                label_q1,
                eq_4,
                label_q2,
                ffhq_noise2,
                eq_7,
                label_q0_q2,
                circle_x0_cp,
                circle_x2_cp,
            ),
            arrow_q0_q2.animate.move_to(circle_x1),
            ReplacementTransform(x2, xt),
            FadeIn(noise),
            run_time=2,
        )

        eq_xt = MathTex(
            r"x_t = x_0 + t \beta \cdot \epsilon", color=WHITE, font_size=32
        ).next_to(arrow_q0_q2, UP, buff=0.5)

        label_q0_t = MathTex(
            r"q(x_t \mid x_0) = \mathcal{N}(",
            r"x_0",
            ",",
            r"t \beta",
            ")",
            color=WHITE,
            font_size=32,
        ).next_to(arrow_q0_q2, DOWN, buff=0.5)

        self.play(Write(eq_xt))

        self.play(Write(label_q0_t))

        self.play(Indicate(label_q0_t[1], color=WHITE))
        self.play(Circumscribe(label_q0_t[1], color=WHITE))

        self.play(Indicate(label_q0_t[3], color=WHITE))
        self.play(Circumscribe(label_q0_t[3], color=WHITE))

        normal = MathTex(r"\mathcal{N}(0, 1)", color=WHITE, font_size=32).next_to(
            label_q0_t, DOWN, buff=1.5
        )

        arrow = Arrow(label_q0_t.get_bottom(), normal.get_top(), color=WHITE, buff=0.1)

        self.play(GrowArrow(arrow))
        self.play(Write(normal))

        redcross = Cross(arrow, color=RED)
        self.play(Create(redcross))
        self.play(FadeOut(arrow, redcross, normal, shift=0.5 * DOWN))

        variance_exploding = Text(
            "The Variance Explodes!", color=WHITE, font_size=32
        ).to_edge(UP, buff=1.0)

        self.play(Write(variance_exploding))

        self.play(ApplyWave(variance_exploding))

        self.play(FadeOut(*self.mobjects))


if __name__ == "__main__":
    scene = Scene2_2()
    scene.render()
