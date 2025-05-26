from manim import *
import numpy as np
import sys

sys.path.append("../")


class Scene2_5(Scene):
    def construct(self):

        tex_to_color_map = {
            r"\theta": BLUE,
        }
        # Show the expression with KL
        self.next_section(skip_animations=False)

        elbo = MathTex(
            r"- \mathbb{E}_{q(x_{1:T} \mid x_0)} \left[ \log \frac{p_{\theta}(x_{0:T})}{q(x_{1:T} \mid x_0)} \right]",
            color=WHITE,
            font_size=32,
        )

        elbo_rect = SurroundingRectangle(elbo, color=WHITE, buff=0.1)
        elbo_label = Tex("Evidence Lower Bound (ELBO)", font_size=32).next_to(
            elbo_rect, UP, buff=0.2
        )

        self.add(
            elbo,
            elbo_rect,
            elbo_label,
        )

        elbo_complete = MathTex(
            r"\mathbb{E}_q \left[",
            r"D_{\mathrm{KL}}( q(x_T \mid x_0) \mid \mid p(x_T))",
            r"+",
            r"\sum_{t > 1} D_{\mathrm{KL}}( q(x_{t-1} \mid x_t, x_0) \mid \mid p_{\theta}(x_{t-1} \mid x_t))",
            r"-",
            r"\log p_{\theta}(x_0 \mid x_1)",
            r"\right]",
            color=WHITE,
            font_size=28,
            tex_to_color_map=tex_to_color_map,
        ).to_edge(UP, buff=0.5)
        self.play(
            FadeOut(elbo_rect),
            FadeOut(elbo_label),
            ReplacementTransform(elbo, elbo_complete),
            run_time=2,
        )

        ddpm_paper_1 = ImageMobject(
            "img/2006.11239v2-13.png", z_index=1
        ).scale_to_fit_width(4)
        ddpm_paper_1.to_edge(DOWN, buff=-1.1 * ddpm_paper_1.height)
        ddpm_paper_2 = ImageMobject("img/2006.11239v2-14.png").scale_to_fit_width(4)
        ddpm_paper_2.to_edge(DOWN, buff=-1.1 * ddpm_paper_1.height).shift(0.2 * LEFT)

        self.play(
            LaggedStart(
                ddpm_paper_1.animate.move_to(ORIGIN),
                ddpm_paper_2.animate.move_to(0.2 * (DOWN + LEFT)),
                lag_ratio=0.2,
            ),
            run_time=2,
        )

        self.play(FadeOut(ddpm_paper_1, ddpm_paper_2), shift=0.5 * DOWN)

        # Explain what KL divergence is
        self.next_section(skip_animations=False)

        ax = (
            Axes(
                x_range=[-4, 4],
                y_range=[0, 1],
                axis_config={"include_tip": False},
                z_index=2,
            )
            .scale(0.65)
            .next_to(elbo_complete, DOWN, buff=0.75)
            .shift(LEFT * 2)
        )

        def gaussian(x, mu, sigma):
            return np.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (
                sigma * np.sqrt(2 * np.pi)
            )

        x = np.linspace(-4, 4, 100)
        g2 = ax.plot(lambda x: gaussian(x, 1, 0.8), color=RED, fill_opacity=0.5)
        g1 = ax.plot(lambda x: gaussian(x, -1, 0.8), color=BLUE, fill_opacity=0.5)
        p_label = (
            MathTex("p", font_size=40, color=RED)
            .next_to(ax, UP, buff=-1.5)
            .shift(RIGHT * 2)
        )
        q_label = (
            MathTex("q", font_size=40, color=BLUE)
            .next_to(ax, UP, buff=-1.5)
            .shift(LEFT * 2)
        )

        kl_label2 = Text("Divergence", font_size=24).next_to(
            ax, RIGHT, buff=1.0, aligned_edge=DOWN
        )
        kl_label = Text("Kullback-Leibler", font_size=24).next_to(
            kl_label2, UP, buff=0.2
        )

        kl_bar = Rectangle(
            height=2,
            width=0.4,
            fill_color=YELLOW,
            stroke_color=YELLOW,
            fill_opacity=1.0,
            sheen_direction=UP,
            sheen_factor=0.7,
        ).next_to(kl_label, UP, buff=0.25)
        kl_rect = SurroundingRectangle(kl_bar, color=WHITE, buff=0.0, z_index=1)
        kl_bar = Rectangle(
            height=1,
            width=0.4,
            fill_color=YELLOW,
            stroke_color=YELLOW,
            fill_opacity=1.0,
            sheen_direction=UP,
            sheen_factor=0.7,
        ).next_to(kl_label, UP, buff=0.25)

        kl_0 = MathTex(r"0", font_size=24, color=WHITE).next_to(
            kl_bar, LEFT, aligned_edge=DOWN, buff=0.2
        )

        kl_1 = MathTex(r"2", font_size=24, color=WHITE).next_to(
            kl_rect, LEFT, aligned_edge=UP, buff=0.2
        )

        self.play(
            LaggedStart(
                Create(ax),
                AnimationGroup(Create(g2), Write(q_label)),
                AnimationGroup(Create(g1), Write(p_label)),
                lag_ratio=0.5,
            ),
            run_time=3,
        )

        self.play(
            FadeIn(kl_bar, kl_label, kl_label2, kl_rect, kl_0, kl_1),
            run_time=1,
        )

        mu = -2

        new_g1 = ax.plot(
            lambda x: gaussian(x, mu, 0.8), color=BLUE, fill_opacity=0.5, z_index=1
        )
        kl_height = abs(mu - 1) / 2  # Simple approximation of KL divergence
        new_bar = Rectangle(
            height=kl_height,
            width=0.4,
            fill_color=YELLOW,
            stroke_color=YELLOW,
            fill_opacity=1.0,
            sheen_direction=UP,
            sheen_factor=0.7,
        ).move_to(kl_bar, aligned_edge=DOWN)
        self.play(
            LaggedStart(
                Transform(g1, new_g1), Transform(kl_bar, new_bar), lag_ratio=0.1
            ),
            run_time=2,
        )

        mu = 0

        new_g1 = ax.plot(
            lambda x: gaussian(x, mu, 0.8), color=BLUE, fill_opacity=0.5, z_index=1
        )
        kl_height = abs(mu - 1) / 2  # Simple approximation of KL divergence
        new_bar = Rectangle(
            height=kl_height,
            width=0.4,
            fill_color=YELLOW,
            stroke_color=YELLOW,
            fill_opacity=1.0,
            sheen_direction=UP,
            sheen_factor=0.7,
        ).move_to(kl_bar, aligned_edge=DOWN)
        self.play(
            LaggedStart(
                Transform(g1, new_g1), Transform(kl_bar, new_bar), lag_ratio=0.1
            ),
            run_time=2,
        )

        mu = 1

        new_g1 = ax.plot(
            lambda x: gaussian(x, mu, 0.8), color=BLUE, fill_opacity=0.5, z_index=1
        )
        kl_height = abs(mu - 1) / 2  # Simple approximation of KL divergence
        new_bar = Rectangle(
            height=kl_height,
            width=0.4,
            fill_color=YELLOW,
            stroke_color=YELLOW,
            fill_opacity=1.0,
            sheen_direction=UP,
            sheen_factor=0.7,
        ).move_to(kl_bar, aligned_edge=DOWN)
        self.play(
            LaggedStart(
                Transform(g1, new_g1), Transform(kl_bar, new_bar), lag_ratio=0.1
            ),
            run_time=2,
        )

        self.play(
            LaggedStart(
                FadeOut(
                    ax,
                    g1,
                    g2,
                    p_label,
                    q_label,
                    kl_bar,
                    kl_label,
                    kl_label2,
                    kl_rect,
                    kl_0,
                    kl_1,
                ),
                elbo_complete.animate.shift(2 * DOWN),
                lag_ratio=0.5,
            ),
            run_time=1.5,
        )

        # Explain why we can ignore the first and last term
        self.next_section(skip_animations=False)

        elbo_simplified = MathTex(
            r"- \mathbb{E}_q \left[",
            r"\sum_{t > 1} D_{\mathrm{KL}}(",
            r"q(x_{t-1} \mid x_t, x_0)",
            r"\mid \mid ",
            r"p_{\theta}(x_{t-1} \mid x_t)",
            r")",
            r"\right]",
            color=WHITE,
            font_size=28,
            tex_to_color_map=tex_to_color_map,
        ).to_edge(UP, buff=0.5)

        rect_1 = SurroundingRectangle(
            elbo_complete[1],
            color=WHITE,
            buff=0.1,
        )
        self.play(Create(rect_1))

        not_depend = MathTex(
            r"\text{Does not depend on} ~", r"\theta", font_size=32
        ).next_to(rect_1, DOWN, buff=0.5)
        not_depend[1].set_color(BLUE)
        self.play(Write(not_depend))

        cross_1 = Cross(
            elbo_complete[1],
            color=RED,
        )

        self.play(LaggedStart(FadeOut(rect_1), Create(cross_1), lag_ratio=0.3))

        rect_2 = SurroundingRectangle(
            elbo_complete[7:10],
            color=WHITE,
            buff=0.1,
        )
        x_0 = ImageMobject("img/ffhq_3.png").scale_to_fit_width(1.5)
        x_1 = (
            ImageMobject("img/ffhq_3_noise_little.png")
            .scale_to_fit_width(1.5)
            .next_to(x_0, LEFT, buff=1.5)
        )
        x_0_rect = SurroundingRectangle(x_0, color=WHITE, buff=0.0)
        x_1_rect = SurroundingRectangle(x_1, color=WHITE, buff=0.0)
        x_0_label = MathTex(r"x_0", font_size=32).next_to(x_0, DOWN, buff=0.2)
        x_1_label = MathTex(r"x_1", font_size=32).next_to(x_1, DOWN, buff=0.2)

        arrow_p = CurvedArrow(
            x_1.get_top() + 0.05 * UP,
            x_0.get_top() + 0.05 * UP,
            color=WHITE,
            angle=-PI / 2,
        )
        arrow_p_label = MathTex(
            r"p_{\theta}(x_0 \mid x_1)",
            font_size=32,
            tex_to_color_map=tex_to_color_map,
        ).next_to(
            arrow_p,
            UP,
            buff=0.2,
        )

        vg = (
            Group(
                x_0,
                x_1,
                x_0_rect,
                x_1_rect,
                x_0_label,
                x_1_label,
                arrow_p,
                arrow_p_label,
            )
            .next_to(rect_2, DOWN, buff=0.75)
            .shift(LEFT * 2)
        )

        self.play(Create(rect_2))

        self.play(
            LaggedStart(
                FadeIn(
                    x_1,
                    x_1_rect,
                    x_1_label,
                ),
                Create(arrow_p),
                Write(arrow_p_label),
                FadeIn(
                    x_0,
                    x_0_rect,
                    x_0_label,
                ),
                lag_ratio=0.5,
            ),
            run_time=3,
        )

        self.play(FadeOut(vg))
        very_small = Tex("Very Small", font_size=32).next_to(rect_2, DOWN, buff=0.5)
        self.play(Write(very_small))

        cross_2 = Cross(
            elbo_complete[7:10],
            color=RED,
        )

        self.play(LaggedStart(FadeOut(rect_2), Create(cross_2), lag_ratio=0.3))

        self.play(
            LaggedStart(
                FadeOut(
                    cross_1,
                    cross_2,
                    elbo_complete[1],
                    elbo_complete[7:10],
                    very_small,
                    not_depend,
                    shift=0.5 * DOWN,
                ),
                ReplacementTransform(
                    VGroup(elbo_complete[0], elbo_complete[2:7], elbo_complete[10:]),
                    elbo_simplified,
                ),
                lag_ratio=0.5,
            ),
            run_time=2,
        )

        self.play(Circumscribe(elbo_simplified[4:7], color=WHITE, run_time=1.5))

        self.play(Circumscribe(elbo_simplified[2], color=WHITE, run_time=1.5))

        self.wait(1)


if __name__ == "__main__":
    scene = Scene2_5()
    scene.render()
