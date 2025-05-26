from manim import *
from manim_voiceover import VoiceoverScene

import numpy as np
import sys

sys.path.append("../")


class Scene2_6(VoiceoverScene, MovingCameraScene):
    def construct(self):

        tex_to_color_map = {
            r"\theta": BLUE,
        }

        # Explain what the q distributions are
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

        self.add(elbo_simplified)

        xt_circle = Circle(radius=0.5, color=WHITE).move_to(RIGHT * 2)
        xt1_circle = Circle(radius=0.5, color=WHITE).move_to(LEFT * 2)
        x0_circle = Circle(radius=0.5, color=WHITE).next_to(xt_circle, DOWN, buff=1.5)
        x0xt_line = DashedLine(
            start=xt_circle.get_bottom(),
            end=x0_circle.get_top(),
            color=WHITE,
        )
        xt_image = (
            ImageMobject("img/ffhq_1_noise_61.png")
            .scale_to_fit_width(1.5)
            .next_to(xt_circle, RIGHT, buff=0.75)
        )
        xt_image_rect = SurroundingRectangle(
            xt_image,
            color=WHITE,
            buff=0.0,
        )
        xt_img_line = DashedLine(
            start=xt_circle.get_right(),
            end=xt_image.get_left(),
            color=WHITE,
            buff=0.0,
        )
        xt1_image = (
            ImageMobject("img/ffhq_1_noise_30.png")
            .scale_to_fit_width(1.5)
            .next_to(xt1_circle, LEFT, buff=0.75)
        )
        xt1_image_rect = SurroundingRectangle(
            xt1_image,
            color=WHITE,
            buff=0.0,
        )
        xt1_img_line = DashedLine(
            start=xt1_circle.get_left(),
            end=xt1_image.get_right(),
            color=WHITE,
            buff=0.0,
        )
        x0_image = (
            ImageMobject("img/ffhq_1.png")
            .scale_to_fit_width(1.5)
            .next_to(x0_circle, RIGHT, buff=0.75)
        )
        x0_image_rect = SurroundingRectangle(
            x0_image,
            color=WHITE,
            buff=0.0,
        )
        x0_img_line = DashedLine(
            start=x0_circle.get_right(),
            end=x0_image.get_left(),
            color=WHITE,
            buff=0.0,
        )

        xt_label = MathTex(
            r"x_t",
            color=WHITE,
            font_size=32,
        ).move_to(xt_circle)

        xt1_label = MathTex(
            r"x_{t-1}",
            color=WHITE,
            font_size=32,
        ).move_to(xt1_circle)
        x0_label = MathTex(
            r"x_0",
            color=WHITE,
            font_size=32,
        ).move_to(x0_circle)

        q_arrow = CurvedArrow(
            start_point=xt1_circle.get_top() + UP * 0.05,
            end_point=xt_circle.get_top() + UP * 0.05,
            color=WHITE,
            angle=-PI / 2,
        )
        q_label = MathTex(
            r"q(x_t \mid x_{t-1})",
            tex_to_color_map=tex_to_color_map,
            color=WHITE,
            font_size=32,
        ).next_to(q_arrow, UP, buff=0.1)
        p_arrow = DashedVMobject(
            CurvedArrow(
                start_point=xt_circle.get_bottom() + DOWN * 0.05,
                end_point=xt1_circle.get_bottom() + DOWN * 0.05,
                color=BLUE,
                angle=-PI / 2,
            )
        )
        p_label = MathTex(
            r"p_{\theta}(x_{t-1} \mid x_t)",
            color=WHITE,
            font_size=32,
            tex_to_color_map=tex_to_color_map,
        ).next_to(p_arrow, DOWN, buff=0.1)
        q_posterior_arrow = CurvedArrow(
            start_point=x0xt_line.get_center() + LEFT * 0.05,
            end_point=xt1_circle.get_bottom() + DOWN * 0.5,
            color=WHITE,
            angle=-2 * PI / 3,
        )
        q_posterior_label = MathTex(
            r"q(x_{t-1} \mid x_t, x_0)",
            color=WHITE,
            font_size=32,
            tex_to_color_map=tex_to_color_map,
        ).next_to(q_posterior_arrow, DOWN, buff=0.1)

        self.play(
            FadeIn(xt1_circle, xt1_image, xt1_image_rect, xt1_label, xt1_img_line)
        )
        self.play(
            LaggedStart(Create(q_arrow), Write(q_label), lag_ratio=0.5),
            run_time=1.5,
        )
        self.play(FadeIn(xt_circle, xt_image, xt_image_rect, xt_label, xt_img_line))

        self.play(
            LaggedStart(Create(p_arrow), Write(p_label), lag_ratio=0.5),
            run_time=1.5,
        )

        self.play(
            LaggedStart(
                Create(x0xt_line),
                FadeIn(x0_circle, x0_image, x0_image_rect, x0_label, x0_img_line),
                lag_ratio=0.7,
            ),
            run_time=1,
        )

        self.play(
            LaggedStart(Create(q_posterior_arrow), Write(q_posterior_label)),
            run_time=1.5,
        )

        self.play(Indicate(VGroup(x0_circle, x0_label), color=WHITE))
        self.play(Indicate(VGroup(x0_circle, x0_label), color=WHITE))

        ax = (
            Axes(
                x_range=[-4, 4],
                y_range=[0, 1],
                axis_config={"include_tip": False, "stroke_width": 1},
                z_index=2,
            )
            .scale(0.3)
            .next_to(q_label, UP, buff=0.4)
        )

        def gaussian(x, mu, sigma):
            return np.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (
                sigma * np.sqrt(2 * np.pi)
            )

        def mixture_gaussian(x, mus, sigmas, weights):
            return sum(
                w * gaussian(x, mu, sigma) for mu, sigma, w in zip(mus, sigmas, weights)
            )

        x = np.linspace(-4, 4, 100)

        # First mixture: two components
        p_plot = ax.plot(
            lambda x: mixture_gaussian(x, [-1.8, -0.3], [0.6, 0.3], [0.7, 0.3]),
            color=BLUE,
            fill_opacity=0.5,
            stroke_width=1,
        )

        # Second mixture: two components
        q_plot = ax.plot(
            lambda x: mixture_gaussian(x, [0.7, 2.1], [0.7, 0.3], [0.4, 0.6]),
            color=RED,
            fill_opacity=0.5,
            stroke_width=1,
        )
        p_plot_label = ax.get_graph_label(
            p_plot,
            label=MathTex(
                r"p_{\theta}(x_{t-1} \mid x_t)",
                tex_to_color_map=tex_to_color_map,
                font_size=10,
            ),
            x_val=-2,
            direction=UP,
            buff=0.5,
            color=WHITE,
        ).shift(0.6 * UP + 0.8 * LEFT)
        q_plot_label = ax.get_graph_label(
            q_plot,
            label=MathTex(
                r"q(x_{t-1} \mid x_t, x_0)",
                tex_to_color_map=tex_to_color_map,
                font_size=10,
            ),
            x_val=2,
            direction=UP,
            buff=2,
            color=WHITE,
        ).shift(0.6 * UP + 0.2 * RIGHT)

        self.play(FadeOut(elbo_simplified))

        # Start the axes small and zoomed out
        self.play(
            LaggedStart(
                FadeIn(ax, p_plot, q_plot),
                self.camera.frame.animate.scale(0.3).move_to(ax),  # Zoom in
                lag_ratio=0.8,
            ),
            run_time=4,
        )

        self.play(Write(p_plot_label))
        self.play(Write(q_plot_label))

        # What shape we choose for the p distributions
        self.next_section(skip_animations=False)

        q_expression = MathTex(
            r"q(x_{t-1} \mid x_t, x_0)",
            r"= \mathcal{N}(\tilde{\mu}_t, \tilde{\beta}_t)",
            color=WHITE,
            font_size=10,
        ).move_to(q_plot_label, aligned_edge=LEFT)

        self.play(Write(q_expression[1]))

        q_gaussian = ax.plot(
            lambda x: gaussian(x, 0.7, 0.5),
            color=RED,
            fill_opacity=0.5,
            stroke_width=1,
        )

        self.play(Transform(q_plot, q_gaussian))

        p_gaussian = ax.plot(
            lambda x: gaussian(x, -1.8, 0.6),
            color=BLUE,
            fill_opacity=0.5,
            stroke_width=1,
        )

        p_expression = MathTex(
            r"p_\theta (x_{t-1} \mid x_t)",
            r"= \mathcal{N}(\mu_\theta, \sigma_\theta)",
            color=WHITE,
            font_size=10,
            tex_to_color_map=tex_to_color_map,
        ).move_to(p_plot_label, aligned_edge=LEFT)

        self.play(
            LaggedStart(
                Transform(p_plot, p_gaussian),
                Write(p_expression[3:]),
                lag_ratio=0.8,
            )
        )

        p_line = DashedLine(
            start=ax.c2p(-1.8, gaussian(-1.8, -1.8, 0.6)),
            end=ax.c2p(-1.8, 0),
            color=BLUE,
            stroke_width=1,
        )
        p_line_label = MathTex(
            r"\mu_\theta",
            color=WHITE,
            font_size=10,
            tex_to_color_map=tex_to_color_map,
        ).next_to(p_line, DOWN, buff=0.1)
        p_line_label.add_updater(lambda m: m.next_to(p_line, DOWN, buff=0.1))

        q_line = DashedLine(
            start=ax.c2p(0.7, gaussian(0.7, 0.7, 0.5)),
            end=ax.c2p(0.7, 0),
            color=RED,
            stroke_width=1,
        )
        q_line_label = MathTex(
            r"\tilde{\mu}_t",
            color=WHITE,
            font_size=10,
            tex_to_color_map=tex_to_color_map,
        ).next_to(q_line, DOWN, buff=0.1)

        self.play(
            LaggedStart(Create(p_line), Write(p_line_label), lag_ratio=0.9),
            run_time=1.5,
        )

        # Create double-ended arrows to show the spread (standard deviation)
        p_spread = DoubleArrow(
            ax.c2p(-2.4, gaussian(-2.4, -1.8, 0.6)),
            ax.c2p(-1.2, gaussian(-1.2, -1.8, 0.6)),
            color=BLUE,
            buff=0,
            stroke_width=1,
            tip_length=0.05,
            max_tip_length_to_length_ratio=0.2,
        )
        p_spread_label = MathTex(
            r"\sigma_\theta",
            color=WHITE,
            font_size=10,
        ).next_to(p_spread, UP, buff=0.1)
        p_spread_label.add_updater(lambda m: m.next_to(p_spread, UP, buff=0.1))

        self.play(
            LaggedStart(Create(p_spread), Write(p_spread_label), lag_ratio=0.9),
            run_time=1.5,
        )

        p_expression_2 = MathTex(
            r"p_\theta (x_{t-1} \mid x_t)",
            r"= \mathcal{N}(\mu_\theta, \sigma_t)",
            color=WHITE,
            font_size=10,
            tex_to_color_map=tex_to_color_map,
        ).move_to(p_plot_label, aligned_edge=LEFT)

        p_spread_label_2 = MathTex(
            r"\sigma_t",
            color=WHITE,
            font_size=10,
            tex_to_color_map=tex_to_color_map,
        ).next_to(p_spread, UP, buff=0.1)

        self.play(
            LaggedStart(
                Transform(p_expression[3:], p_expression_2[3:]),
                Transform(p_spread_label, p_spread_label_2),
                lag_ratio=0.5,
            ),
            run_time=2,
        )

        self.play(Circumscribe(p_line_label, color=WHITE, stroke_width=1))

        self.play(
            LaggedStart(Create(q_line), Write(q_line_label), lag_ratio=0.9),
            run_time=1.5,
        )

        new_p_gaussian = ax.plot(
            lambda x: gaussian(x, -0.4, 0.45),
            color=BLUE,
            fill_opacity=0.5,
            stroke_width=1,
        )

        self.play(
            FadeOut(p_spread, p_spread_label),
            Transform(p_plot, new_p_gaussian),
            p_line.animate.put_start_and_end_on(
                start=ax.c2p(-0.4, gaussian(-0.4, -0.4, 0.45)),
                end=ax.c2p(-0.4, 0),
            ),
            run_time=2,
        )

        kl_expression = MathTex(
            r"D_{\mathrm{KL}}(q(x_{t-1} \mid x_t, x_0) \mid \mid p_\theta (x_{t-1} \mid x_t)) = \frac{1}{2 \sigma_t^2}} \| \tilde{\mu}_t - \mu_\theta \|^2",
            color=WHITE,
            font_size=10,
            tex_to_color_map=tex_to_color_map,
        ).next_to(p_expression, UP, buff=0.05, aligned_edge=LEFT)

        self.play(Write(kl_expression))

        # Show that we try to match all distributions at the same time
        self.next_section(skip_animations=False)

        self.remove(
            x0_image,
            x0_circle,
            x0_image_rect,
            x0_label,
            x0_img_line,
            xt1_image,
            xt1_image_rect,
            xt1_img_line,
            xt_image,
            xt_image_rect,
            xt_img_line,
            q_posterior_arrow,
            q_posterior_label,
            x0xt_line,
        )

        xt2_circle = Circle(radius=0.5, color=WHITE).next_to(xt1_circle, LEFT, buff=3)
        xt2_label = MathTex(
            r"x_{t-2}",
            color=WHITE,
            font_size=32,
        ).move_to(xt2_circle)

        xt11_circle = Circle(radius=0.5, color=WHITE).next_to(xt_circle, RIGHT, buff=3)
        xt11_label = MathTex(
            r"x_{t+1}",
            color=WHITE,
            font_size=32,
        ).move_to(xt11_circle)

        q_arrow_t2 = CurvedArrow(
            start_point=xt2_circle.get_top() + UP * 0.05,
            end_point=xt1_circle.get_top() + UP * 0.05,
            color=WHITE,
            angle=-PI / 2,
        )

        q_label_t2 = MathTex(
            r"q(x_{t-1} \mid x_{t-2})",
            tex_to_color_map=tex_to_color_map,
            color=WHITE,
            font_size=32,
        ).next_to(q_arrow_t2, UP, buff=0.1)

        q_arrow_t1 = CurvedArrow(
            start_point=xt_circle.get_top() + UP * 0.05,
            end_point=xt11_circle.get_top() + UP * 0.05,
            color=WHITE,
            angle=-PI / 2,
        )
        q_label_t1 = MathTex(
            r"q(x_{t-1} \mid x_{t+1})",
            tex_to_color_map=tex_to_color_map,
            color=WHITE,
            font_size=32,
        ).next_to(q_arrow_t1, UP, buff=0.1)

        p_arrow_t2 = DashedVMobject(
            CurvedArrow(
                start_point=xt1_circle.get_bottom() + DOWN * 0.05,
                end_point=xt2_circle.get_bottom() + DOWN * 0.05,
                color=BLUE,
                angle=-PI / 2,
            )
        )
        p_label_t2 = MathTex(
            r"p_{\theta}(x_{t-2} \mid x_{t-1})",
            color=WHITE,
            font_size=32,
            tex_to_color_map=tex_to_color_map,
        ).next_to(p_arrow_t2, DOWN, buff=0.1)
        p_arrow_t1 = DashedVMobject(
            CurvedArrow(
                start_point=xt11_circle.get_bottom() + DOWN * 0.05,
                end_point=xt_circle.get_bottom() + DOWN * 0.05,
                color=BLUE,
                angle=-PI / 2,
            )
        )
        p_label_t1 = MathTex(
            r"p_{\theta}(x_{t} \mid x_{t+1})",
            color=WHITE,
            font_size=32,
            tex_to_color_map=tex_to_color_map,
        ).next_to(p_arrow_t1, DOWN, buff=0.1)

        ax2 = (
            Axes(
                x_range=[-4, 4],
                y_range=[0, 1],
                axis_config={"include_tip": False, "stroke_width": 1},
                z_index=2,
            )
            .scale(0.3)
            .next_to(q_label_t1, UP, buff=0.4)
        )

        p2_plot = ax2.plot(
            lambda x: gaussian(x, 1.2, 0.7),
            color=BLUE,
            fill_opacity=0.5,
            stroke_width=1,
        )
        q2_plot = ax2.plot(
            lambda x: gaussian(x, -0.2, 0.5),
            color=RED,
            fill_opacity=0.5,
            stroke_width=1,
        )

        ax3 = (
            Axes(
                x_range=[-4, 4],
                y_range=[0, 1],
                axis_config={"include_tip": False, "stroke_width": 1},
                z_index=2,
            )
            .scale(0.3)
            .next_to(q_label_t2, UP, buff=0.4)
        )
        p3_plot = ax3.plot(
            lambda x: gaussian(x, -0.4, 0.55),
            color=BLUE,
            fill_opacity=0.5,
            stroke_width=1,
        )
        q3_plot = ax3.plot(
            lambda x: gaussian(x, 1.2, 0.6),
            color=RED,
            fill_opacity=0.5,
            stroke_width=1,
        )

        self.add(
            xt2_circle,
            xt11_circle,
            xt2_label,
            xt11_label,
            q_arrow_t2,
            q_label_t2,
            q_arrow_t1,
            q_label_t1,
            p_arrow_t2,
            p_label_t2,
            p_arrow_t1,
            p_label_t1,
            ax2,
            p2_plot,
            q2_plot,
            ax3,
            p3_plot,
            q3_plot,
        )

        # Reset camera position
        self.play(
            self.camera.frame.animate.scale(1 / 0.3).move_to(ORIGIN),
            FadeOut(
                kl_expression,
                p_expression[3:],
                p_plot_label,
                q_plot_label,
                p_line_label,
                q_line_label,
                q_line,
                p_line,
                q_expression,
            ),
            run_time=2.5,
        )

        # Animate 3 times, each time bringing p closer to q
        for i in range(3):
            # For each iteration, move p means closer to q means using weighted average
            weight = (i + 1) / 3
            p3_mean = -0.4 * (1 - weight) + 1.2 * weight  # Move from -0.4 to 1.2
            p2_mean = 1.2 * (1 - weight) + (-0.2) * weight  # Move from 1.2 to -0.2
            p_mean = -0.4 * (1 - weight) + 0.7 * weight  # Move from -0.4 to 0.7

            p3_plot_new = ax3.plot(
                lambda x: gaussian(x, p3_mean, 0.55),
                color=BLUE,
                fill_opacity=0.5,
                stroke_width=1,
            )
            p2_plot_new = ax2.plot(
                lambda x: gaussian(x, p2_mean, 0.7),
                color=BLUE,
                fill_opacity=0.5,
                stroke_width=1,
            )
            p_plot_new = ax.plot(
                lambda x: gaussian(x, p_mean, 0.6),
                color=BLUE,
                fill_opacity=0.5,
                stroke_width=1,
            )

            self.play(
                LaggedStart(
                    Transform(p3_plot, p3_plot_new),
                    Transform(p2_plot, p2_plot_new),
                    Transform(p_plot, p_plot_new),
                    lag_ratio=0.1,
                    run_time=2,
                )
            )

        self.play(FadeOut(*self.mobjects, shift=0.5 * DOWN))

        self.wait(1)


if __name__ == "__main__":
    scene = Scene2_6()
    scene.render()
