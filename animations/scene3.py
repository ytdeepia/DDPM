from manim import *

import numpy as np
import sys

sys.path.append("../")
from src.kde_density import KDECotours


# Define bimodal distribution function
def bimodal_dist(x):
    dist1 = np.exp(-0.5 * ((x - 2) / 0.5) ** 2) / (0.5 * np.sqrt(2 * np.pi))
    dist2 = np.exp(-0.5 * ((x + 1) / 0.5) ** 2) / (0.5 * np.sqrt(2 * np.pi))
    return 0.6 * dist1 + 0.4 * dist2


def apply_ddpm_step(dist_func, t, beta=0.02):
    def transformed_dist(x):
        alpha_bar_t = (1 - beta) ** t
        return np.mean(
            [
                dist_func(
                    (
                        (x - np.sqrt(1 - alpha_bar_t) * np.random.normal())
                        / np.sqrt(alpha_bar_t)
                    )
                )
                for _ in range(1000)
            ]
        ) / np.sqrt(alpha_bar_t)

    return transformed_dist


class Scene2_3(Scene):
    def construct(self):

        # Explain the objective
        self.next_section(skip_animations=False)

        q_xt_x0 = MathTex(r"q(x_t|x_0)", font_size=32)
        normal = MathTex(r"\mathcal{N}(0, 1)", font_size=32)

        VGroup(q_xt_x0, normal).arrange(RIGHT, buff=2).move_to(ORIGIN)
        arrow1 = Arrow(
            q_xt_x0.get_right(),
            normal.get_left(),
            buff=0.25,
        )
        t_inf = MathTex(r"t \to \infty", font_size=20).next_to(arrow1, DOWN, buff=0.1)

        title_obj = Tex("Objective", font_size=42).next_to(
            VGroup(q_xt_x0, normal, t_inf, arrow1), UP, buff=0.5
        )

        title_obj_ul = Underline(title_obj)

        self.play(LaggedStart(Write(title_obj), Create(title_obj_ul)))

        self.play(
            Write(q_xt_x0),
        )

        self.play(
            LaggedStart(
                GrowArrow(arrow1),
                FadeIn(t_inf),
            )
        )
        self.play(
            Write(normal),
        )

        rect_objective = SurroundingRectangle(
            VGroup(q_xt_x0, normal, t_inf, arrow1, title_obj, title_obj_ul),
            color=WHITE,
            buff=0.1,
        )

        objective = VGroup(
            q_xt_x0, normal, t_inf, arrow1, title_obj, title_obj_ul, rect_objective
        )

        self.play(Create(rect_objective))
        self.play(objective.animate.to_edge(UP, buff=0.2))

        circle_xt1 = Circle(radius=0.5, color=WHITE).shift(LEFT * 3)
        circle_xt = Circle(radius=0.5, color=WHITE).shift(RIGHT * 3)

        xt1 = MathTex(r"x_{t-1}", font_size=32).move_to(circle_xt1)
        xt = MathTex(r"x_t", font_size=32).move_to(circle_xt)

        arrow_xt1_xt = Arrow(
            circle_xt1.get_right(),
            circle_xt.get_left(),
        )

        self.play(FadeIn(circle_xt1, xt1))
        self.play(GrowArrow(arrow_xt1_xt))
        self.play(FadeIn(circle_xt, xt))

        # Define the iterates
        self.next_section(skip_animations=False)

        q_xt_xt1 = MathTex(
            r"q(x_t|x_{t-1}) = ",
            r"\sqrt{1 - \beta} ~",
            r" x_{t-1} + \beta \cdot \epsilon",
            font_size=32,
        ).next_to(arrow_xt1_xt, UP, buff=0.5)

        rect_empty = SurroundingRectangle(
            q_xt_xt1[1],
            color=WHITE,
            buff=0.0,
        )
        qmark = Tex("?", font_size=32).move_to(rect_empty)

        self.play(
            LaggedStart(
                Write(q_xt_xt1[0]),
                AnimationGroup(Create(rect_empty), Create(qmark)),
                Write(q_xt_xt1[2]),
                lag_ratio=0.95,
            ),
            run_time=2.5,
        )

        self.play(ReplacementTransform(VGroup(rect_empty, qmark), q_xt_xt1[1]))

        q_xt_x0_exp = MathTex(
            r"q(x_t|x_0) =",
            r"\sqrt{\bar{\alpha}_t} ~ x_0",
            r"+",
            r"(1 - \bar{\alpha}_t)",
            r"\cdot \epsilon",
            font_size=32,
        ).next_to(arrow_xt1_xt, UP, buff=0.5)

        a_t = MathTex(r"\bar{\alpha}_t =", r"(1 - \beta)^{t}", font_size=32).next_to(
            arrow_xt1_xt, DOWN, buff=0.5
        )

        x0 = MathTex(r"x_0", font_size=32).move_to(circle_xt1)

        self.play(FadeOut(q_xt_xt1))
        self.play(ReplacementTransform(xt1, x0))
        self.play(Write(q_xt_x0_exp))

        self.play(Write(a_t))

        # Check that the process does indeed converge to the normal distribution
        self.next_section(skip_animations=False)

        self.play(Circumscribe(a_t, color=WHITE))

        self.play(Circumscribe(q_xt_x0_exp[1], color=WHITE), buff=0.02)
        self.play(Circumscribe(q_xt_x0_exp[3], color=WHITE), buff=0.02)

        chain = VGroup(circle_xt1, circle_xt, arrow_xt1_xt, x0, xt, q_xt_x0_exp, a_t)

        self.play(FadeOut(objective), VGroup(chain).animate.to_edge(UP, buff=0.2))

        plot1d = (
            Axes(
                x_range=[-5, 5, 1],
                y_range=[0, 0.5, 0.1],
                axis_config={"color": WHITE},
            )
            .scale_to_fit_width(6)
            .shift(DOWN)
        )

        # Plot the distribution
        plot_bimodal = plot1d.plot(lambda x: bimodal_dist(x), color=BLUE)

        self.play(Create(plot1d), run_time=2)
        self.play(Create(plot_bimodal), run_time=2)

        # Iterate over 100 DDPM steps
        current_plot = plot_bimodal
        beta = 0.02

        label = MathTex(r"p(x)", font_size=32, color=BLUE).next_to(
            plot1d, UL, buff=-0.8
        )
        self.play(Write(label))

        # Plot initial DDPM step
        current_plot = plot_bimodal
        new_dist = apply_ddpm_step(bimodal_dist, 1, beta)
        new_plot = plot1d.plot(lambda x: new_dist(x), color=RED)

        self.play(
            ReplacementTransform(current_plot, new_plot),
            Transform(
                label,
                MathTex("q(x_1|x_0)", font_size=32, color=RED).next_to(
                    plot1d, UL, buff=-0.8
                ),
            ),
            run_time=1.5,
        )

        current_plot = new_plot

        # Iterate over 150 DDPM steps
        for i in range(10, 150, 10):
            new_dist = apply_ddpm_step(bimodal_dist, i, beta)
            new_plot = plot1d.plot(lambda x: new_dist(x), color=RED)

            self.play(
                ReplacementTransform(current_plot, new_plot),
                Transform(
                    label,
                    MathTex(f"q(x_{{{i}}}|x_0)", font_size=32, color=RED).next_to(
                        plot1d, UL, buff=-0.8
                    ),
                ),
                run_time=0.5,
            )
            current_plot = new_plot

        # Show the final distribution
        self.next_section(skip_animations=False)

        plot_bimodal = plot1d.plot(lambda x: bimodal_dist(x), color=BLUE)
        self.play(
            ReplacementTransform(current_plot, plot_bimodal),
            Transform(
                label,
                MathTex("p(x)", font_size=32, color=BLUE).next_to(
                    plot1d, UL, buff=-0.8
                ),
            ),
            run_time=2,
        )

        current_plot = plot_bimodal
        new_dist = apply_ddpm_step(bimodal_dist, 200, beta)
        new_plot = plot1d.plot(lambda x: new_dist(x), color=RED)

        self.play(
            ReplacementTransform(current_plot, new_plot),
            Transform(
                label,
                MathTex("q(x_{200}|x_0)", font_size=32, color=RED).next_to(
                    plot1d, UL, buff=-0.8
                ),
            ),
            run_time=2,
        )

        ddpm_paper = ImageMobject(
            "img/ddpm_1.png",
        ).scale_to_fit_width(6)
        ddpm_paper.to_edge(UP, buff=-1.2 * ddpm_paper.height)
        self.add(ddpm_paper)

        self.play(ddpm_paper.animate.move_to(ORIGIN))

        self.play(FadeOut(ddpm_paper, shift=0.5 * UP))

        self.play(
            LaggedStart(
                Uncreate(new_plot),
                FadeOut(label, plot1d),
                lag_ratio=0.5,
            ),
            run_time=2,
        )

        self.play(chain.animate.move_to(ORIGIN))

        # Talk about varying beta
        self.next_section(skip_animations=False)

        self.play(Circumscribe(a_t, color=WHITE), run_time=1.5)

        a_t_updated = MathTex(
            r"\bar{\alpha}_t = ",
            r"\prod_{i=1}^{t} (1 - \beta_i)",
            font_size=32,
        ).move_to(a_t, aligned_edge=LEFT)

        self.play(Transform(a_t[1], a_t_updated[1]), run_time=2)

        self.play(FadeOut(*self.mobjects, shift=0.5 * DOWN))


if __name__ == "__main__":
    scene = Scene2_3()
    scene.render()
