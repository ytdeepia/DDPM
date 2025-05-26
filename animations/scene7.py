from manim import *
import sys

sys.path.append("../")


class Scene2_7(MovingCameraScene):
    def construct(self):

        # KL into L2
        self.next_section(skip_animations=False)

        tex_to_color_map = {
            r"\theta": BLUE,
        }

        elbo_kl = MathTex(
            r"- \mathbb{E}_q \left[ \sum_{t > 1} D_{\mathrm{KL}}(q(x_{t-1} \mid x_t, x_0) \mid \mid p_\theta (x_{t-1} \mid x_t)) \right]",
            color=WHITE,
            font_size=28,
            tex_to_color_map=tex_to_color_map,
        )
        elbo_kl_cp = elbo_kl.copy()
        elbo_kl_cp.shift(2 * UP)
        elbo_l2 = MathTex(
            r"\mathbb{E}_q \left[ \sum_{t > 1} \frac{1}{2 \sigma_t^2} \|",
            r"\tilde{\mu}_t",
            r"-",
            r"\mu_\theta(x_t, t)",
            r" \|^2 \right]",
            color=WHITE,
            font_size=28,
            tex_to_color_map=tex_to_color_map,
        )

        arrow_elbo = Arrow(
            elbo_kl_cp.get_bottom(), elbo_l2.get_top(), buff=0.1, color=WHITE
        )

        self.play(Write(elbo_kl))
        self.wait(2)
        self.play(
            LaggedStart(
                elbo_kl.animate.shift(2 * UP), GrowArrow(arrow_elbo), lag_ratio=0.7
            ),
            run_time=2,
        )
        self.play(Write(elbo_l2))

        self.play(Circumscribe(elbo_l2[3:6], color=WHITE))
        self.play(Circumscribe(elbo_l2[1], color=WHITE))

        self.play(FadeOut(elbo_kl, arrow_elbo))

        mutilde_expression = MathTex(
            r"\tilde{\mu}_t = \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t} (1- \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t",
            color=WHITE,
            font_size=28,
            tex_to_color_map=tex_to_color_map,
        ).next_to(elbo_l2, UP, buff=1.5)

        # Reparameterization trick
        self.next_section(skip_animations=False)

        mutilde_expression_cp = mutilde_expression.copy()

        reparam = MathTex(
            r"x_t = \sqrt{\alpha_t} x_0 + \sqrt{1 - \alpha_t} \epsilon",
            color=WHITE,
            font_size=28,
            tex_to_color_map=tex_to_color_map,
        )
        VGroup(
            mutilde_expression_cp,
            reparam,
        ).arrange(
            RIGHT, buff=1.0
        ).next_to(elbo_l2, UP, buff=1.5)

        self.play(Write(mutilde_expression))

        x0_img = (
            ImageMobject(
                "./img/ffhq_4.png",
            )
            .scale_to_fit_width(1.5)
            .next_to(mutilde_expression, LEFT, buff=0.5)
        )
        x0_rect = SurroundingRectangle(x0_img, color=WHITE, buff=0.0)
        x0_label = MathTex(
            r"x_0",
            color=WHITE,
            font_size=28,
            tex_to_color_map=tex_to_color_map,
        ).next_to(x0_img, DOWN, buff=0.1)

        xt_img = (
            ImageMobject(
                "./img/ffhq_4_noisy.png",
            )
            .scale_to_fit_width(1.5)
            .next_to(mutilde_expression, RIGHT, buff=0.5)
        )
        xt_rect = SurroundingRectangle(xt_img, color=WHITE, buff=0.0)
        xt_label = MathTex(
            r"x_t",
            color=WHITE,
            font_size=28,
            tex_to_color_map=tex_to_color_map,
        ).next_to(xt_img, DOWN, buff=0.1)

        self.play(FadeIn(x0_img, x0_rect, x0_label))
        self.play(FadeIn(xt_img, xt_rect, xt_label))

        self.play(FadeOut(x0_img, x0_rect, x0_label, xt_img, xt_rect, xt_label))

        self.play(
            LaggedStart(
                mutilde_expression.animate.move_to(mutilde_expression_cp.get_center()),
                Write(reparam),
                lag_ratio=0.9,
            ),
            run_time=1.2,
        )

        reparam_2 = MathTex(
            r"x_0 = \frac{1}{\sqrt{\alpha_t}} (x_t - \sqrt{1 - \alpha_t} \epsilon)",
            color=WHITE,
            font_size=28,
            tex_to_color_map=tex_to_color_map,
        ).move_to(reparam.get_center())

        self.play(Transform(reparam, reparam_2))

        mutilde_reparam = MathTex(
            r"\tilde{\mu}_t = \frac{1}{\sqrt{\alpha}_t} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon \right)",
            color=WHITE,
            font_size=28,
            tex_to_color_map=tex_to_color_map,
        ).next_to(elbo_l2, UP, buff=1.5)

        self.play(
            ReplacementTransform(VGroup(mutilde_expression, reparam), mutilde_reparam)
        )

        xt_img = (
            ImageMobject(
                "./img/ffhq_4_noisy.png",
            )
            .scale_to_fit_width(1.5)
            .next_to(mutilde_reparam, LEFT, buff=0.5)
        )
        xt_rect = SurroundingRectangle(xt_img, color=WHITE, buff=0.0)
        xt_label = MathTex(
            r"x_t",
            color=WHITE,
            font_size=28,
            tex_to_color_map=tex_to_color_map,
        ).next_to(xt_img, DOWN, buff=0.1)

        noise_img = (
            ImageMobject(
                "./img/pure_noise_61.png",
            )
            .scale_to_fit_width(1.5)
            .next_to(mutilde_reparam, RIGHT, buff=0.5)
        )
        noise_rect = SurroundingRectangle(noise_img, color=WHITE, buff=0.0)
        noise_label = MathTex(
            r"\epsilon",
            color=WHITE,
            font_size=28,
            tex_to_color_map=tex_to_color_map,
        ).next_to(noise_img, DOWN, buff=0.1)
        self.play(FadeIn(xt_img, xt_rect, xt_label, noise_img, noise_rect, noise_label))

        self.play(
            FadeOut(xt_img, xt_rect, xt_label, noise_img, noise_rect, noise_label)
        )

        mutheta_reparam = MathTex(
            r"\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha}_t} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)",
            color=WHITE,
            font_size=28,
            tex_to_color_map=tex_to_color_map,
        )

        mutilde_reparam_cp = mutilde_reparam.copy()

        VGroup(
            mutilde_reparam_cp,
            mutheta_reparam,
        ).arrange(
            RIGHT, buff=1.0
        ).next_to(elbo_l2, UP, buff=1.5)
        self.play(
            LaggedStart(
                mutilde_reparam.animate.move_to(mutilde_reparam_cp.get_center()),
                Write(mutheta_reparam),
                lag_ratio=0.7,
            ),
            run_time=2,
        )

        epsilon_loss = MathTex(
            r"\mathcal{L} =  \mathbb{E}_q \left[ \sum_{t > 1} \frac{\beta_t^2}{2 \sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)} \| \epsilon - \epsilon_\theta (x_t, t) \|^2 \right]",
            color=WHITE,
            font_size=28,
            tex_to_color_map=tex_to_color_map,
        ).move_to(elbo_l2)
        self.play(Transform(elbo_l2, epsilon_loss))

        self.wait(3)
        final_loss = Tex("Final loss function", color=WHITE, font_size=40).next_to(
            elbo_l2, UP, buff=1.5
        )

        self.play(FadeOut(mutilde_reparam, mutheta_reparam))
        self.play(Write(final_loss))
        self.play(ShowPassingFlash(Underline(final_loss), color=WHITE))

        self.play(FadeOut(*self.mobjects))

        # Recap of all the steps
        self.next_section(skip_animations=False)

        title = Tex("Recap of all the steps", color=WHITE, font_size=40).to_edge(
            UP, buff=0.25
        )

        self.play(Write(title))

        step1_rect = Rectangle(width=4, height=2.5).set_color(WHITE)
        step2_rect = Rectangle(width=4, height=2.5).set_color(WHITE)
        step3_rect = Rectangle(width=4, height=2.5).set_color(WHITE)
        step4_rect = Rectangle(width=4, height=2.5).set_color(WHITE)
        step5_rect = Rectangle(width=4, height=2.5).set_color(WHITE)
        step6_rect = Rectangle(width=4, height=2.5).set_color(WHITE)

        all_steps = (
            VGroup(
                step1_rect,
                step2_rect,
                step3_rect,
                step6_rect,
                step5_rect,
                step4_rect,
            )
            .arrange_in_grid(
                n_rows=2,
                n_cols=3,
                buff=1,
            )
            .next_to(title, DOWN, buff=0.5)
        )
        step1_circ = Square(side_length=0.4, color=WHITE).next_to(
            step1_rect.get_corner(UL), DR, buff=0
        )
        step2_circ = Square(side_length=0.4, color=WHITE).next_to(
            step2_rect.get_corner(UL), DR, buff=0
        )
        step3_circ = Square(side_length=0.4, color=WHITE).next_to(
            step3_rect.get_corner(UL), DR, buff=0
        )
        step4_circ = Square(side_length=0.4, color=WHITE).next_to(
            step4_rect.get_corner(UL), DR, buff=0
        )
        step5_circ = Square(side_length=0.4, color=WHITE).next_to(
            step5_rect.get_corner(UL), DR, buff=0
        )
        step6_circ = Square(side_length=0.4, color=WHITE).next_to(
            step6_rect.get_corner(UL), DR, buff=0
        )

        step1_num = Text("1", font_size=20).move_to(step1_circ)
        step2_num = Text("2", font_size=20).move_to(step2_circ)
        step3_num = Text("3", font_size=20).move_to(step3_circ)
        step4_num = Text("4", font_size=20).move_to(step4_circ)
        step5_num = Text("5", font_size=20).move_to(step5_circ)
        step6_num = Text("6", font_size=20).move_to(step6_circ)

        numbers = VGroup(
            step1_num,
            step2_num,
            step3_num,
            step4_num,
            step5_num,
            step6_num,
            step1_circ,
            step2_circ,
            step3_circ,
            step4_circ,
            step5_circ,
            step6_circ,
        )
        arrows = VGroup(
            Arrow(step1_rect.get_right(), step2_rect.get_left(), color=WHITE),
            Arrow(step2_rect.get_right(), step3_rect.get_left(), color=WHITE),
            Arrow(step3_rect.get_bottom(), step4_rect.get_top(), color=WHITE),
            Arrow(step4_rect.get_left(), step5_rect.get_right(), color=WHITE),
            Arrow(step5_rect.get_left(), step6_rect.get_right(), color=WHITE),
        )

        step1_title = Tex(
            "Negative log-likelihood",
            color=WHITE,
            font_size=24,
        ).next_to(step1_rect, UP, buff=-0.3)

        eq_step1 = MathTex(
            r"- \log p_\theta(x_0) = - \log \int p_\theta(x_0, x_t) dx_t",
            color=WHITE,
            font_size=18,
            tex_to_color_map=tex_to_color_map,
        ).move_to(step1_rect.get_center())

        step2_title = Tex(
            "Evidence lower bound",
            color=WHITE,
            font_size=24,
        ).next_to(step2_rect, UP, buff=-0.3)
        eq_step2 = MathTex(
            r"- \log p_\theta(x_0) \leq \mathbb{E}_q \left[ - \log \frac{p_\theta(x_0, x_t)}{q(x_t \mid x_0)} \right]",
            color=WHITE,
            font_size=18,
        ).move_to(step2_rect.get_center())

        step3_title = Tex(
            "Rewrite the ELBO",
            color=WHITE,
            font_size=24,
        ).next_to(step3_rect, UP, buff=-0.3)
        eq_step3_line1 = MathTex(
            r"\mathbb{E}_q \left[ D_{\mathrm{KL}}(q(x_T \mid x_0) \mid \mid p(x_T))",
            color=WHITE,
            font_size=18,
            tex_to_color_map=tex_to_color_map,
        )

        eq_step3_line2 = MathTex(
            r"+ \sum_{t > 1} D_{\mathrm{KL}}(q(x_{t-1} \mid x_t, x_0) \mid \mid p_\theta(x_{t-1} \mid x_t))",
            color=WHITE,
            font_size=18,
            tex_to_color_map=tex_to_color_map,
        )

        eq_step3_line3 = MathTex(
            r"- \log p_\theta(x_0 \mid x_1) \right]",
            color=WHITE,
            font_size=18,
            tex_to_color_map=tex_to_color_map,
        )

        eq_step3 = (
            VGroup(eq_step3_line1, eq_step3_line2, eq_step3_line3)
            .arrange(DOWN, aligned_edge=LEFT, buff=0.15)
            .move_to(step3_rect.get_center())
        )

        step4_title = Tex(
            "Simplify the ELBO",
            color=WHITE,
            font_size=24,
        ).next_to(step4_rect, UP, buff=-0.3)
        eq_step4 = MathTex(
            r"\mathbb{E}_q \left[",
            r"\sum_{t > 1} D_{\mathrm{KL}}( q(x_{t-1} \mid x_t, x_0) \mid \mid p_\theta(x_{t-1} \mid x_t))",
            r"\right]",
            color=WHITE,
            font_size=18,
            tex_to_color_map=tex_to_color_map,
        ).move_to(step4_rect.get_center())

        step5_title = Tex(
            "Use Gaussians",
            color=WHITE,
            font_size=24,
        ).next_to(step5_rect, UP, buff=-0.3)
        eq_step5 = MathTex(
            r"\mathbb{E}_q \left[ \frac{1}{2 \sigma_t^2} \sum_{t > 1} \| \tilde{\mu}_t - \mu_\theta(x_t, t) \|^2 \right]",
            color=WHITE,
            font_size=18,
            tex_to_color_map=tex_to_color_map,
        ).move_to(step5_rect.get_center())

        step6_title = Tex(
            "Reparameterization trick",
            color=WHITE,
            font_size=24,
        ).next_to(step6_rect, UP, buff=-0.3)
        eq_step6 = MathTex(
            r"\mathbb{E}_q \left[",
            r"\sum_{t > 1}",
            r"\frac{\beta_t^2}{2 \sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)}",
            r"\| \epsilon - \epsilon_\theta (x_t, t) \|^2 \right]",
            color=WHITE,
            font_size=18,
            tex_to_color_map=tex_to_color_map,
        ).move_to(step6_rect.get_center())

        titles = VGroup(
            step1_title,
            step2_title,
            step3_title,
            step4_title,
            step5_title,
            step6_title,
        )

        self.play(
            LaggedStart(
                Create(step1_rect),
                AnimationGroup(Create(step1_circ), FadeIn(step1_num)),
                Write(step1_title),
                lag_ratio=0.7,
            )
        )

        self.play(FadeIn(eq_step1))

        self.play(GrowArrow(arrows[0]))

        self.play(
            LaggedStart(
                Create(step2_rect),
                AnimationGroup(Create(step2_circ), FadeIn(step2_num)),
                Write(step2_title),
                lag_ratio=0.7,
            )
        )

        self.play(FadeIn(eq_step2))

        self.play(GrowArrow(arrows[1]))

        self.play(
            LaggedStart(
                Create(step3_rect),
                AnimationGroup(Create(step3_circ), FadeIn(step3_num)),
                Write(step3_title),
                lag_ratio=0.7,
            )
        )
        self.play(FadeIn(eq_step3))

        self.play(GrowArrow(arrows[2]))

        self.play(
            LaggedStart(
                Create(step4_rect),
                AnimationGroup(Create(step4_circ), FadeIn(step4_num)),
                Write(step4_title),
                lag_ratio=0.7,
            )
        )
        self.play(FadeIn(eq_step4))

        self.play(GrowArrow(arrows[3]))

        self.play(
            LaggedStart(
                Create(step5_rect),
                AnimationGroup(Create(step5_circ), FadeIn(step5_num)),
                Write(step5_title),
                lag_ratio=0.7,
            )
        )
        self.play(FadeIn(eq_step5))

        self.play(GrowArrow(arrows[4]))

        self.play(
            LaggedStart(
                Create(step6_rect),
                AnimationGroup(Create(step6_circ), FadeIn(step6_num)),
                Write(step6_title),
                lag_ratio=0.7,
            )
        )
        self.play(FadeIn(eq_step6))

        # Very Last simplification
        self.next_section(skip_animations=False)

        self.play(
            eq_step6.animate.move_to(ORIGIN).scale(1.55),
            FadeOut(
                all_steps,
                numbers,
                titles,
                eq_step1,
                eq_step2,
                eq_step3,
                eq_step4,
                eq_step5,
                arrows,
                title,
            ),
        )

        cross_sum = Cross(eq_step6[1], color=WHITE, stroke_width=2)
        self.play(Create(cross_sum))

        eq_step6_expectation = MathTex(
            r"\mathbb{E}_{q, t} \left[ \frac{\beta_t^2}{2 \sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)} \| \epsilon - \epsilon_\theta (x_t, t) \|^2 \right]",
            color=WHITE,
            font_size=28,
            tex_to_color_map=tex_to_color_map,
        ).move_to(eq_step6)
        self.play(
            LaggedStart(FadeOut(cross_sum), Transform(eq_step6, eq_step6_expectation))
        )

        self.play(FadeOut(eq_step6, shift=0.5 * DOWN))

        self.wait(1)


if __name__ == "__main__":
    scene = Scene2_7()
    scene.render()
