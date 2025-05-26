from manim import *
import sys

sys.path.append("../")


class Scene2_4(Scene):
    def construct(self):
        # Show the reverse distributions
        self.next_section(skip_animations=False)

        img_x0 = ImageMobject("./img/ffhq_3.png").scale_to_fit_width(2)
        img_x0_rect = SurroundingRectangle(img_x0, color=WHITE, buff=0.0)
        img_xt_1 = ImageMobject("./img/ffhq_3_noise_61.png").scale_to_fit_width(2)
        img_xt_1_rect = SurroundingRectangle(img_xt_1, color=WHITE, buff=0.0)
        img_xt = ImageMobject("./img/ffhq_3_noise_102.png").scale_to_fit_width(2)
        img_xt_rect = SurroundingRectangle(img_xt, color=WHITE, buff=0.0)
        img_xT = ImageMobject("./img/pure_noise.png").scale_to_fit_width(2)
        img_xT_rect = SurroundingRectangle(img_xT, color=WHITE, buff=0.0)

        circle_x0 = Circle(radius=0.5, color=WHITE)
        x0 = MathTex(r"x_0", color=WHITE, font_size=32).move_to(circle_x0)
        circle_xt_1 = Circle(radius=0.5, color=WHITE)
        x1 = MathTex(r"x_{t-1}", color=WHITE, font_size=32).move_to(circle_xt_1)
        circle_xt = Circle(radius=0.5, color=WHITE)
        xt = MathTex(r"x_t", color=WHITE, font_size=32).move_to(circle_xt)
        circle_xT = Circle(radius=0.5, color=WHITE)
        xT = MathTex(r"x_T", color=WHITE, font_size=32).move_to(circle_xT)

        x0g = VGroup(circle_x0, x0)
        x1g = VGroup(circle_xt_1, x1)
        xtg = VGroup(circle_xt, xt)
        xTg = VGroup(circle_xT, xT)

        Group(x0g, x1g, xtg, xTg).arrange(RIGHT, buff=2).move_to(ORIGIN)

        Group(img_x0, img_x0_rect).next_to(x0g, UP, buff=1.5)
        Group(img_xt_1, img_xt_1_rect).next_to(x1g, UP, buff=1.5)
        Group(img_xt, img_xt_rect).next_to(xtg, UP, buff=1.5)
        Group(img_xT, img_xT_rect).next_to(xTg, UP, buff=1.5)

        linex0 = DashedLine(x0g.get_top(), img_x0.get_bottom(), color=WHITE)
        linext1 = DashedLine(x1g.get_top(), img_xt_1.get_bottom(), color=WHITE)
        linext = DashedLine(xtg.get_top(), img_xt.get_bottom(), color=WHITE)
        linexT = DashedLine(xTg.get_top(), img_xT.get_bottom(), color=WHITE)

        self.play(
            LaggedStart(
                AnimationGroup(Create(circle_x0), Write(x0)),
                Create(linex0),
                FadeIn(img_x0, img_x0_rect),
                lag_ratio=0.4,
            ),
            run_time=1.5,
        )

        dots_x0_xt = dots = (
            VGroup(*[Dot(color=WHITE, z_index=4, radius=0.025) for _ in range(3)])
            .arrange(RIGHT, buff=0.05)
            .move_to((x0g.get_right() + x1g.get_left()) / 2)
        )
        dots_xt1_xT = (
            VGroup(*[Dot(color=WHITE, z_index=4, radius=0.025) for _ in range(3)])
            .arrange(RIGHT, buff=0.05)
            .move_to((xtg.get_right() + xTg.get_left()) / 2)
        )

        self.play(LaggedStartMap(Create, dots_x0_xt), run_time=0.5)
        self.play(
            LaggedStart(
                AnimationGroup(Create(circle_xt_1), Write(x1)),
                Create(linext1),
                FadeIn(img_xt_1, img_xt_1_rect),
            ),
            run_time=1.5,
        )

        arrow_q_xt1_xt = CurvedArrow(
            x1g.get_top() + 0.1 * UP,
            xtg.get_top() + 0.1 * UP,
            angle=-PI / 2,
            color=WHITE,
        )

        label_q_xt1_xt = MathTex(
            r"q(x_t \mid x_{t-1})", color=WHITE, font_size=32
        ).next_to(arrow_q_xt1_xt, UP, buff=0.2)

        self.play(
            LaggedStart(Create(arrow_q_xt1_xt), Write(label_q_xt1_xt), lag_ratio=0.6),
            run_time=1.5,
        )

        self.play(
            LaggedStart(
                AnimationGroup(Create(circle_xt), Write(xt)),
                Create(linext),
                FadeIn(img_xt, img_xt_rect),
            )
        )

        self.play(LaggedStartMap(Create, dots_xt1_xT), run_time=0.5)
        self.play(
            LaggedStart(
                AnimationGroup(Create(circle_xT), Write(xT)),
                Create(linexT),
                FadeIn(img_xT, img_xT_rect),
            ),
            run_time=1.5,
        )
        # Show the reverse distributions
        self.next_section(skip_animations=False)

        img_x0 = ImageMobject("./img/ffhq_3.png").scale_to_fit_width(2)
        img_x0_rect = SurroundingRectangle(img_x0, color=WHITE, buff=0.0)
        img_xt_1 = ImageMobject("./img/ffhq_3_noise_61.png").scale_to_fit_width(2)
        img_xt_1_rect = SurroundingRectangle(img_xt_1, color=WHITE, buff=0.0)
        img_xt = ImageMobject("./img/ffhq_3_noise_102.png").scale_to_fit_width(2)
        img_xt_rect = SurroundingRectangle(img_xt, color=WHITE, buff=0.0)
        img_xT = ImageMobject("./img/pure_noise.png").scale_to_fit_width(2)
        img_xT_rect = SurroundingRectangle(img_xT, color=WHITE, buff=0.0)

        circle_x0 = Circle(radius=0.5, color=WHITE)
        x0 = MathTex(r"x_0", color=WHITE, font_size=32).move_to(circle_x0)
        circle_xt_1 = Circle(radius=0.5, color=WHITE)
        x1 = MathTex(r"x_{t-1}", color=WHITE, font_size=32).move_to(circle_xt_1)
        circle_xt = Circle(radius=0.5, color=WHITE)
        xt = MathTex(r"x_t", color=WHITE, font_size=32).move_to(circle_xt)
        circle_xT = Circle(radius=0.5, color=WHITE)
        xT = MathTex(r"x_T", color=WHITE, font_size=32).move_to(circle_xT)

        x0g = VGroup(circle_x0, x0)
        x1g = VGroup(circle_xt_1, x1)
        xtg = VGroup(circle_xt, xt)
        xTg = VGroup(circle_xT, xT)

        Group(x0g, x1g, xtg, xTg).arrange(RIGHT, buff=2).move_to(ORIGIN)

        Group(img_x0, img_x0_rect).next_to(x0g, UP, buff=1.5)
        Group(img_xt_1, img_xt_1_rect).next_to(x1g, UP, buff=1.5)
        Group(img_xt, img_xt_rect).next_to(xtg, UP, buff=1.5)
        Group(img_xT, img_xT_rect).next_to(xTg, UP, buff=1.5)

        linex0 = DashedLine(x0g.get_top(), img_x0.get_bottom(), color=WHITE)
        linext1 = DashedLine(x1g.get_top(), img_xt_1.get_bottom(), color=WHITE)
        linext = DashedLine(xtg.get_top(), img_xt.get_bottom(), color=WHITE)
        linexT = DashedLine(xTg.get_top(), img_xT.get_bottom(), color=WHITE)

        self.play(
            LaggedStart(
                AnimationGroup(Create(circle_x0), Write(x0)),
                Create(linex0),
                FadeIn(img_x0, img_x0_rect),
                lag_ratio=0.4,
            ),
            run_time=1.5,
        )

        dots_x0_xt = dots = (
            VGroup(*[Dot(color=WHITE, z_index=4, radius=0.025) for _ in range(3)])
            .arrange(RIGHT, buff=0.05)
            .move_to((x0g.get_right() + x1g.get_left()) / 2)
        )
        dots_xt1_xT = (
            VGroup(*[Dot(color=WHITE, z_index=4, radius=0.025) for _ in range(3)])
            .arrange(RIGHT, buff=0.05)
            .move_to((xtg.get_right() + xTg.get_left()) / 2)
        )

        self.play(LaggedStartMap(Create, dots_x0_xt), run_time=0.5)
        self.play(
            LaggedStart(
                AnimationGroup(Create(circle_xt_1), Write(x1)),
                Create(linext1),
                FadeIn(img_xt_1, img_xt_1_rect),
            ),
            run_time=1.5,
        )

        arrow_q_xt1_xt = CurvedArrow(
            x1g.get_top() + 0.1 * UP,
            xtg.get_top() + 0.1 * UP,
            angle=-PI / 2,
            color=WHITE,
        )

        label_q_xt1_xt = MathTex(
            r"q(x_t \mid x_{t-1})", color=WHITE, font_size=32
        ).next_to(arrow_q_xt1_xt, UP, buff=0.2)

        self.play(
            LaggedStart(Create(arrow_q_xt1_xt), Write(label_q_xt1_xt), lag_ratio=0.6),
            run_time=1.5,
        )

        self.play(
            LaggedStart(
                AnimationGroup(Create(circle_xt), Write(xt)),
                Create(linext),
                FadeIn(img_xt, img_xt_rect),
            )
        )

        self.play(LaggedStartMap(Create, dots_xt1_xT), run_time=0.5)
        self.play(
            LaggedStart(
                AnimationGroup(Create(circle_xT), Write(xT)),
                Create(linexT),
                FadeIn(img_xT, img_xT_rect),
            ),
            run_time=1.5,
        )

        arrow_p_xt_xt1 = CurvedArrow(
            xtg.get_bottom() + 0.1 * DOWN,
            x1g.get_bottom() + 0.1 * DOWN,
            angle=-PI / 2,
            color=WHITE,
        )

        label_p_xt_xt1 = MathTex(
            r"p", r"(x_{t-1} \mid x_t)", color=WHITE, font_size=32
        ).next_to(arrow_p_xt_xt1, DOWN, buff=0.2)

        self.play(
            LaggedStart(Create(arrow_p_xt_xt1), Write(label_p_xt_xt1), lag_ratio=0.6),
            run_time=1.5,
        )

        label_p_theta_xt_xt1 = MathTex(
            r"p_", r"{\theta}", r"(x_{t-1} \mid x_t)", color=WHITE, font_size=32
        ).next_to(arrow_p_xt_xt1, DOWN, buff=0.2)
        label_p_theta_xt_xt1[1].set_color(BLUE)

        self.play(
            Transform(
                label_p_xt_xt1,
                label_p_theta_xt_xt1,
                run_time=1,
            )
        )

        self.play(
            FadeOut(
                linex0,
                linext1,
                linext,
                linexT,
                img_x0,
                img_x0_rect,
                img_xt_1,
                img_xt_1_rect,
                img_xt,
                img_xt_rect,
                img_xT,
                img_xT_rect,
            ),
            run_time=0.7,
        )

        train_txt = Tex(
            "How do we train the ", "neural network", " ?", font_size=42
        ).to_edge(UP, buff=0.2)
        train_txt[1].set_color(BLUE)

        self.play(Write(train_txt))

        self.play(FadeOut(train_txt), run_time=0.7)

        objective = Tex(
            "We minimize the negative log-likelihood", font_size=42
        ).to_edge(UP, buff=0.2)

        nll = MathTex(
            r"-\log p_", r"{\theta}", r"(x_0)", color=WHITE, font_size=32
        ).next_to(objective, DOWN, buff=0.5)
        nll[1].set_color(BLUE)
        rect_nll = SurroundingRectangle(nll, color=WHITE, buff=0.1)

        self.play(Write(objective))
        self.play(LaggedStart(FadeIn(nll), Create(rect_nll), buff=0.5), run_time=1.5)

        self.play(
            FadeOut(
                x0g,
                x1g,
                xtg,
                xTg,
                dots_x0_xt,
                dots_xt1_xT,
                arrow_p_xt_xt1,
                arrow_q_xt1_xt,
                label_q_xt1_xt,
                label_p_xt_xt1,
                shift=0.5 * DOWN,
            ),
            run_time=1.25,
        )

        # Clear up the notations
        self.next_section(skip_animations=False)

        joint_q_bayes = (
            MathTex(
                r"q(x_1, \ldots, x_T \mid x_0)",
                r"= q(x_1 \mid x_0) ~ q(x_2 \mid x_1, x_0) ~ \ldots ~ q(x_T \mid x_{T-1}, \ldots, x_0)",
                color=WHITE,
                font_size=32,
            )
            .to_edge(LEFT, buff=1.5)
            .shift(UP)
        )
        self.play(Write(joint_q_bayes[0]), run_time=0.5)

        self.play(Write(joint_q_bayes[1]))

        q_bayes_simple = MathTex(
            r"q(x_1, \ldots, x_T \mid x_0) = q(x_1 \mid x_0) ~ q(x_2 \mid x_1) ~ \ldots ~ q(x_T \mid x_{T-1})",
            color=WHITE,
            font_size=32,
        ).next_to(joint_q_bayes, DOWN, buff=0.75, aligned_edge=LEFT)

        self.play(Write(q_bayes_simple), run_time=1.5)
        q_bayes_simple_1 = MathTex(
            r"q(x_{1:T} \mid x_0) =",
            r"q(x_1 \mid x_0) ~ q(x_2 \mid x_1) ~ \ldots ~ q(x_T \mid x_{T-1})",
            color=WHITE,
            font_size=32,
        ).next_to(q_bayes_simple, DOWN, buff=0.75, aligned_edge=LEFT)

        self.play(Write(q_bayes_simple_1))

        q_bayes_simple_2 = MathTex(
            r"\prod_{t=1}^T q(x_t \mid x_{t-1})",
            color=WHITE,
            font_size=32,
        ).next_to(q_bayes_simple_1[0], RIGHT)

        self.play(Transform(q_bayes_simple_1[1], q_bayes_simple_2))

        self.play(
            q_bayes_simple_1.animate.to_edge(LEFT, buff=1.5).shift(2 * UP),
            FadeOut(joint_q_bayes, q_bayes_simple),
        )

        joint_p = (
            MathTex(
                r"p_{",
                r"\theta",
                r"}(x_{0:T}) = p_{\theta}(x_T) \prod_{t=1}^T p_{\theta}(x_{t-1} \mid x_t)",
                color=WHITE,
                font_size=32,
            )
            .next_to(q_bayes_simple_1, RIGHT)
            .to_edge(RIGHT, buff=1.5)
        )
        joint_p[1].set_color(BLUE)

        self.play(Write(joint_p))

        label_q = Tex("Complete forward process", font_size=32).next_to(
            q_bayes_simple_1, DOWN, buff=0.5
        )
        label_p = Tex("Complete reverse process", font_size=32).next_to(
            joint_p, DOWN, buff=0.5
        )

        self.play(Write(label_q))
        self.play(Write(label_p))

        # Show that the likelihood is intractable
        self.next_section(skip_animations=False)

        marginalize = MathTex(
            r"- \log p_",
            r"\theta",
            r"(x_0)",
            r"= - \log \int p_{",
            r"\theta",
            r"}(x_{0:T})",
            r"dx_{1:T}",
            color=WHITE,
            font_size=32,
        ).shift(1.75 * UP)
        marginalize[1].set_color(BLUE)
        marginalize[4].set_color(BLUE)

        self.play(
            FadeOut(objective, rect_nll, label_q, label_p),
            VGroup(joint_p, q_bayes_simple_1).animate.to_edge(UP, buff=0.5),
            nll.animate.move_to(marginalize, aligned_edge=LEFT),
            run_time=1.5,
        )
        self.play(Write(marginalize[3:]))

        l_ellipse = Ellipse(width=2, height=3, color=WHITE).shift(DOWN + 2 * LEFT)
        r_ellipse = Ellipse(width=2, height=3, color=WHITE).shift(DOWN + 2 * RIGHT)

        l_ellipse_title = Tex("Images", font_size=32).next_to(l_ellipse, UP, buff=0.2)
        r_ellipse_title = Tex("Noise", font_size=32).next_to(r_ellipse, UP, buff=0.2)
        l_dot = Dot(l_ellipse.get_center(), color=WHITE)
        r_dot = Dot(r_ellipse.get_center(), color=WHITE)

        # Create the paths
        paths = []
        angles = [-PI / 2, -PI / 3, 0, PI / 3, PI / 2]
        for angle in angles:
            path = DashedVMobject(
                ArcBetweenPoints(
                    r_dot.get_center(), l_dot.get_center(), angle=angle, color=BLUE
                )
            )
            paths.append(path)

        self.play(
            Create(l_ellipse),
        )
        self.play(Write(l_ellipse_title))
        self.play(
            Create(r_ellipse),
        )
        self.play(Write(r_ellipse_title))

        l_dot_label = MathTex(r"x_0", color=WHITE, font_size=32).next_to(
            l_dot, UP + LEFT, buff=0.2
        )
        r_dot_label = MathTex(r"x_T", color=WHITE, font_size=32).next_to(
            r_dot, UP + RIGHT, buff=0.2
        )
        self.play(
            FadeIn(l_dot, l_dot_label),
        )

        self.play(
            FadeIn(r_dot, r_dot_label),
        )
        self.play(Create(paths[0]))

        self.play(Circumscribe(marginalize[6], color=WHITE))
        self.play(LaggedStartMap(Create, paths[1:]), run_time=2)

        self.play(LaggedStartMap(FadeOut, paths, shift=0.5 * DOWN), run_time=2)

        self.play(
            FadeOut(l_ellipse, l_ellipse_title, l_dot, l_dot_label),
            FadeOut(r_ellipse, r_ellipse_title, r_dot, r_dot_label),
        )

        # Derive the ELBO surrogate
        self.next_section(skip_animations=False)

        marginalize_trick = MathTex(
            r"- \log p_{\theta}(x_0) = - \log \int q(x_{1:T} \mid x_0) \frac{p_{\theta}(x_{0:T})}{q(x_{1:T} \mid x_0)} dx_{1:T}",
            color=WHITE,
            font_size=32,
        ).next_to(marginalize, DOWN, buff=0.5, aligned_edge=LEFT)

        self.play(Write(marginalize_trick), run_time=1)

        expectactions = MathTex(
            r"- \log p_{\theta}(x_0) = - \log \mathbb{E}_{q(x_{1:T} \mid x_0)} \left[ \frac{p_{\theta}(x_{0:T})}{q(x_{1:T} \mid x_0)} \right] ",
            color=WHITE,
            font_size=32,
        ).next_to(marginalize_trick, DOWN, buff=0.5, aligned_edge=LEFT)

        self.play(Write(expectactions), run_time=1)

        jensen = MathTex(
            r"- \log p_{\theta}(x_0) \leq ",
            r"- \mathbb{E}_{q(x_{1:T} \mid x_0)} \left[ \log \frac{p_{\theta}(x_{0:T})}{q(x_{1:T} \mid x_0)} \right] ",
            color=WHITE,
            font_size=32,
        ).next_to(expectactions, DOWN, buff=0.5, aligned_edge=LEFT)

        self.play(Write(jensen), run_time=1)

        elbo = jensen[1]

        self.play(
            FadeOut(nll, marginalize[3:], marginalize_trick, expectactions, jensen[0]),
            elbo.animate.move_to(ORIGIN),
        )

        elbo_rect = SurroundingRectangle(elbo, color=WHITE, buff=0.1)
        elbo_label = Tex("Evidence Lower Bound (ELBO)", font_size=32).next_to(
            elbo_rect, UP, buff=0.2
        )

        self.play(Create(elbo_rect))
        self.play(Write(elbo_label))

        self.play(FadeOut(q_bayes_simple_1, joint_p))

        self.wait(1)


if __name__ == "__main__":
    scene = Scene2_4()
    scene.render()
