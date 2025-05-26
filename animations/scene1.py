from manim import *
import sys

sys.path.append("../")
from src.kde_density import KDEContours


class Scene2_1(Scene):
    def construct(self):

        # Show the papers
        self.next_section(skip_animations=False)

        # Load images
        ddpm_p0 = ImageMobject("img/ddpm_0.png", z_index=2).scale_to_fit_width(3)
        ddpm_p1 = ImageMobject("img/ddpm_1.png", z_index=1).scale_to_fit_width(3)
        ddpm_p2 = ImageMobject("img/ddpm_2.png", z_index=0).scale_to_fit_width(3)

        diffusion_p0 = ImageMobject(
            "img/diffusion_0.png", z_index=2
        ).scale_to_fit_width(3)
        diffusion_p1 = ImageMobject(
            "img/diffusion_1.png", z_index=1
        ).scale_to_fit_width(3)
        diffusion_p2 = ImageMobject(
            "img/diffusion_2.png", z_index=0
        ).scale_to_fit_width(3)

        ddpm_p0.move_to(ORIGIN)
        ddpm_p1.next_to(ddpm_p0, RIGHT * 0.5 + DOWN * 0.5, aligned_edge=UP + LEFT)
        ddpm_p2.next_to(ddpm_p1, RIGHT * 0.5 + DOWN * 0.5, aligned_edge=UP + LEFT)

        start_y_offset = 8  # Large enough to be off-screen
        ddpm_p0.shift(DOWN * start_y_offset)
        ddpm_p1.shift(DOWN * start_y_offset)
        ddpm_p2.shift(DOWN * start_y_offset)

        self.play(
            LaggedStart(
                ddpm_p0.animate.shift(UP * start_y_offset),
                ddpm_p1.animate.shift(UP * start_y_offset),
                ddpm_p2.animate.shift(UP * start_y_offset),
                lag_ratio=0.5,
                run_time=2.0,
            )
        )

        ddpm_date = Tex("2020").scale(0.5).next_to(ddpm_p0, UP)
        ddpm_title = Tex(
            "Denoising Diffusion Probabilistic Models", font_size=32
        ).next_to(ddpm_date, UP, buff=0.25)
        ddpm_title_ul = Underline(ddpm_title)

        self.play(LaggedStart(Write(ddpm_title), Create(ddpm_title_ul), lag_ratio=0.5))
        self.play(Write(ddpm_date))

        # Move DDPM paper to the left and add diffusion paper
        ddpm_paper = Group(
            ddpm_p0, ddpm_p1, ddpm_p2, ddpm_title, ddpm_title_ul, ddpm_date
        )

        self.play(ddpm_paper.animate.shift(4 * LEFT))

        diffusion_p0.move_to(4 * RIGHT)
        diffusion_p1.next_to(
            diffusion_p0, RIGHT * 0.5 + DOWN * 0.5, aligned_edge=UP + LEFT
        )
        diffusion_p2.next_to(
            diffusion_p1, RIGHT * 0.5 + DOWN * 0.5, aligned_edge=UP + LEFT
        )
        diffusion_p0.shift(DOWN * start_y_offset)
        diffusion_p1.shift(DOWN * start_y_offset)
        diffusion_p2.shift(DOWN * start_y_offset)

        self.play(
            LaggedStart(
                diffusion_p0.animate.shift(UP * start_y_offset),
                diffusion_p1.animate.shift(UP * start_y_offset),
                diffusion_p2.animate.shift(UP * start_y_offset),
                lag_ratio=0.5,
                run_time=2.0,
            )
        )

        diffusion_date = Tex("2015").scale(0.5).next_to(diffusion_p0, UP)
        diffusion_title_2 = Tex("Nonequilibrium Thermodynamics", font_size=32).next_to(
            diffusion_date, UP, buff=0.25
        )
        diffusion_title_1 = Tex(
            "Deep Unsupervised Learning using", font_size=32
        ).next_to(diffusion_title_2, UP, buff=0.25)
        diffusion_title_ul_1 = Underline(diffusion_title_1)
        diffusion_title_ul_2 = Underline(diffusion_title_2)

        self.play(
            LaggedStart(
                Write(VGroup(diffusion_title_1, diffusion_title_2)),
                Create(diffusion_title_ul_1),
                Create(diffusion_title_ul_2),
                lag_ratio=0.5,
            )
        )
        self.play(Write(diffusion_date))

        # Highlight DDPM paper and fade out diffusion paper
        diffusion_paper = Group(
            diffusion_p0,
            diffusion_p1,
            diffusion_p2,
            diffusion_title_1,
            diffusion_title_2,
            diffusion_title_ul_1,
            diffusion_title_ul_2,
            diffusion_date,
        )

        self.play(Circumscribe(ddpm_paper, shape=Rectangle, color=WHITE, run_time=2))
        self.play(FadeOut(diffusion_paper))

        citation_number = Tex("20 000+ citations", font_size=32).move_to(3 * RIGHT)
        self.play(Write(citation_number))
        self.play(FadeOut(citation_number, shift=0.5 * DOWN))

        # Main contributions
        self.next_section(skip_animations=False)

        contribution_title = Tex("Main contributions", font_size=32).to_edge(
            RIGHT, buff=3.0
        )
        contribution_title_ul = Underline(contribution_title)
        # Create contribution items directly without loops
        contrib_item1 = Tex(
            r"$\rightarrow$ ", "State-of-the-art", " in image generation", color=WHITE
        ).scale(0.5)
        contrib_item2 = Tex(r"$\rightarrow$ Simpler training objective").scale(0.5)
        contrib_item3 = Tex(
            r"$\rightarrow$ Connection to denoising score matching"
        ).scale(0.5)

        contrib_item1.next_to(contribution_title, DOWN, buff=1.0)
        contrib_item2.next_to(contrib_item1, DOWN, aligned_edge=LEFT, buff=0.75)
        contrib_item3.next_to(contrib_item2, DOWN, aligned_edge=LEFT, buff=0.75)

        contrib_items = VGroup(contrib_item1, contrib_item2, contrib_item3)

        contrib_box = SurroundingRectangle(
            VGroup(contribution_title, contribution_title_ul, contrib_items),
            color=WHITE,
            buff=0.3,
            corner_radius=0.1,
        )

        contrib_vgroup = VGroup(
            contribution_title,
            contribution_title_ul,
            contrib_item1,
            contrib_item2,
            contrib_item3,
            contrib_box,
        ).move_to((config.frame_width / 4) * RIGHT)

        self.play(
            LaggedStart(
                Write(contribution_title), Create(contribution_title_ul), lag_ratio=0.5
            )
        )

        self.play(LaggedStart(*[Write(item) for item in contrib_items], lag_ratio=0.5))
        self.play(Create(contrib_box))

        contributions = Group(
            contribution_title, contribution_title_ul, contrib_items, contrib_box
        )

        self.play(FadeOut(contributions, ddpm_paper), shift=0.5 * RIGHT)

        # Data distribution
        self.next_section(skip_animations=False)

        pdf = KDEContours().scale_to_fit_width(5)
        pdf_title = MathTex("p(x)").next_to(pdf, UP, buff=0.25)

        self.play(Create(pdf), Write(pdf_title))

        # Individual image loading and positioning
        img0 = (
            ImageMobject("img/ffhq_0.png")
            .scale_to_fit_width(1.5)
            .next_to(pdf, LEFT, buff=0.2)
        )
        rect0 = SurroundingRectangle(img0, color=WHITE, buff=0.0, z_index=1)
        img1 = (
            ImageMobject("img/ffhq_1.png")
            .scale_to_fit_width(1.5)
            .next_to(pdf, RIGHT, buff=0.2)
        )
        rect1 = SurroundingRectangle(img1, color=WHITE, buff=0.0, z_index=1)
        img2 = (
            ImageMobject("img/ffhq_2.png")
            .scale_to_fit_width(1.5)
            .next_to(pdf, RIGHT + DOWN, buff=-1)
        )
        rect2 = SurroundingRectangle(img2, color=WHITE, buff=0.0, z_index=1)
        dot1 = Dot([-0.8, 0.5, 0], color=WHITE)
        dot2 = Dot([0.5, 0.2, 0], color=WHITE)
        dot3 = Dot([0.0, -0.7, 0], color=WHITE)

        line1 = Line(start=dot1, end=img0.get_right(), color=WHITE)
        line2 = Line(start=dot2, end=img1.get_left(), color=WHITE)
        line3 = Line(start=dot3, end=img2.get_left(), color=WHITE)

        self.play(
            LaggedStart(
                Create(dot1),
                Create(line1),
                Create(rect0),
                FadeIn(img0),
                lag_ratio=0.5,
                run_time=2,
            )
        )
        self.play(
            LaggedStart(
                Create(dot2),
                Create(line2),
                Create(rect1),
                FadeIn(img1),
                lag_ratio=0.5,
                run_time=2,
            )
        )
        self.play(
            LaggedStart(
                Create(dot3),
                Create(line3),
                Create(rect2),
                FadeIn(img2),
                lag_ratio=0.5,
                run_time=2,
            )
        )

        pdf_images = Group(
            pdf,
            pdf_title,
            img0,
            img1,
            img2,
            rect0,
            rect1,
            rect2,
            dot1,
            dot2,
            dot3,
            line1,
            line2,
            line3,
        )
        self.play(pdf_images.animate.scale(0.7).to_edge(LEFT, buff=0.5))

        # Add exponential distributions and question mark
        formula_pos = pdf_images.get_right() + RIGHT * 1.5
        eq_start = MathTex("p(x)", "= ", font_size=32).move_to(formula_pos)
        eq1 = MathTex(
            r"p(x)",
            "=",
            r"\frac{1}{Z} \exp\left(-\sum_{i,j} \alpha_{ij} x_i x_j - \sum_k \beta_k x_k^3\right)",
            font_size=32,
        ).move_to(eq_start.get_left(), aligned_edge=LEFT)
        eq2 = MathTex(
            r"p(x) ",
            "=",
            r"\frac{1}{Z} \exp\left(-\frac{1}{2}x^\top \Sigma^{-1} x - \sum_i \lambda_i |x_i|^\gamma\right)",
            font_size=32,
        ).move_to(eq_start.get_left(), aligned_edge=LEFT)
        question = MathTex("p(x)", "=", "\; ?", font_size=32).move_to(
            eq_start.get_left(), aligned_edge=LEFT
        )
        question[0].set_color(BLUE)

        self.play(Write(eq_start))
        self.play(FadeIn(eq1[2], shift=0.5 * DOWN))
        # Add red cross for eq1
        cross1 = Cross(eq1[2], stroke_color=RED, stroke_width=6)
        self.play(Create(cross1))
        self.play(
            LaggedStart(
                AnimationGroup(FadeOut(eq1[2]), FadeOut(cross1)),
                FadeIn(eq2[2], shift=0.5 * DOWN),
                lag_ratio=0.5,
            )
        )

        # Add red cross for eq2
        cross2 = Cross(eq2[2], stroke_color=RED, stroke_width=6)
        self.play(Create(cross2))
        self.play(
            LaggedStart(
                AnimationGroup(FadeOut(eq2[2]), FadeOut(cross2)),
                FadeIn(question[2], shift=0.5 * DOWN),
                lag_ratio=0.5,
            )
        )

        # Progressive degradation / restoration
        self.next_section(skip_animations=False)

        # Load noisy versions of the image
        img0_noise30 = ImageMobject("img/ffhq_0_noise_30.png").scale_to_fit_width(
            2 * img0.width
        )
        img0_noise60 = ImageMobject("img/ffhq_0_noise_60.png").scale_to_fit_width(
            2 * img0.width
        )
        pure_noise = ImageMobject("img/pure_noise.png").scale_to_fit_width(
            2 * img0.width
        )

        temp_group = Group(img0, img0_noise30, img0_noise60, pure_noise).copy()
        temp_group[0].scale(2)
        temp_group.arrange(RIGHT, buff=1).move_to(ORIGIN)

        img0_pos = temp_group[0].get_center()
        img0_noise30_pos = temp_group[1].get_center()
        img0_noise60_pos = temp_group[2].get_center()
        pure_noise_pos = temp_group[3].get_center()

        self.play(
            LaggedStart(
                AnimationGroup(
                    FadeOut(question, eq_start, pdf_title),
                    FadeOut(pdf, dot1, dot2, dot3, line1, line2, line3),
                    FadeOut(img1, img2, rect1, rect2),
                ),
                Group(img0, rect0).animate.scale(2).move_to(img0_pos),
                lag_ratio=0.5,
            ),
            run_time=2,
        )

        rect0.set_z_index(3)
        img0.set_z_index(2)

        rect_noise30 = SurroundingRectangle(img0_noise30, color=WHITE, buff=0.0)
        rect_noise60 = SurroundingRectangle(img0_noise60, color=WHITE, buff=0.0)
        rect_pure_noise = SurroundingRectangle(pure_noise, color=WHITE, buff=0.0)

        # Set z-indices for all images and rectangles
        img0_noise30.set_z_index(2)
        rect_noise30.set_z_index(3)
        img0_noise60.set_z_index(2)
        rect_noise60.set_z_index(3)
        pure_noise.set_z_index(2)
        rect_pure_noise.set_z_index(3)

        Group(img0_noise30, rect_noise30).move_to(img0_pos)
        Group(img0_noise60, rect_noise60).move_to(img0_pos)
        Group(pure_noise, rect_pure_noise).move_to(img0_pos)

        # Add the noisy images (already handled their z-indices above)
        img0_noise30.z_index = -1
        img0_noise60.z_index = -1
        pure_noise.z_index = -1

        self.add(
            img0_noise30,
            rect_noise30,
            img0_noise60,
            rect_noise60,
            pure_noise,
            rect_pure_noise,
        )

        arrow1 = Arrow(
            temp_group[0].get_right(),
            temp_group[1].get_left(),
            color=WHITE,
            buff=0.1,
            z_index=4,
        )
        arrow2 = Arrow(
            temp_group[1].get_right(),
            temp_group[2].get_left(),
            color=WHITE,
            buff=0.1,
            z_index=4,
        )
        dots = (
            VGroup(*[Dot(color=WHITE, z_index=4, radius=0.025) for _ in range(3)])
            .arrange(RIGHT, buff=0.05)
            .move_to((img0_noise60_pos + pure_noise_pos) / 2)
        )
        # Add labels under each image
        label_x0 = MathTex("x_0", color=WHITE, font_size=32).next_to(
            temp_group[0], DOWN, buff=0.2
        )
        label_x1 = MathTex("x_1", color=WHITE, font_size=32).next_to(
            temp_group[1], DOWN, buff=0.2
        )
        label_x2 = MathTex("x_2", color=WHITE, font_size=32).next_to(
            temp_group[2], DOWN, buff=0.2
        )
        label_xT = MathTex("x_T", color=WHITE, font_size=32).next_to(
            temp_group[3], DOWN, buff=0.2
        )

        self.play(Write(label_x0))
        self.play(GrowArrow(arrow1))
        self.play(
            Group(img0_noise30, rect_noise30).animate.move_to(img0_noise30_pos),
            Write(label_x1),
        )

        self.play(GrowArrow(arrow2))
        self.play(
            Group(img0_noise60, rect_noise60).animate.move_to(img0_noise60_pos),
            Write(label_x2),
        )
        Group(pure_noise, rect_pure_noise).move_to(pure_noise_pos)
        self.play(
            LaggedStart(
                FadeIn(dots),
                FadeIn(Group(pure_noise, rect_pure_noise)),
                Write(label_xT),
                lag_ratio=0.5,
            )
        )

        # Add reversed curved arrows (noise to clean)
        reversed_arrow1 = CurvedArrow(
            pure_noise.get_top(),
            img0_noise60.get_top(),
            color=WHITE,
        )
        reversed_arrow2 = CurvedArrow(
            img0_noise60.get_top(),
            img0_noise30.get_top(),
            color=WHITE,
        )
        reversed_arrow3 = CurvedArrow(
            img0_noise30.get_top(),
            img0.get_top(),
            color=WHITE,
        )

        label_arrow1 = MathTex(r"s_{\theta} (x_T)", color=WHITE, font_size=32).next_to(
            reversed_arrow1, UP, buff=0.2
        )
        label_arrow2 = MathTex(r"s_{\theta} (x_2)", color=WHITE, font_size=32).next_to(
            reversed_arrow2, UP, buff=0.2
        )
        label_arrow3 = MathTex(r"s_{\theta} (x_1)", color=WHITE, font_size=32).next_to(
            reversed_arrow3, UP, buff=0.2
        )

        self.play(Create(reversed_arrow1))
        self.play(Write(label_arrow1))
        self.play(Create(reversed_arrow2))
        self.play(Write(label_arrow2))
        self.play(Create(reversed_arrow3))
        self.play(Write(label_arrow3))

        self.play(FadeOut(*self.mobjects, shift=0.5 * DOWN))

        self.wait(1)


if __name__ == "__main__":
    scene = Scene2_1()
    scene.render()
