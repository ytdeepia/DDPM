from manim import *
import sys

sys.path.append("../")


class Scene2_9(Scene):
    def construct(self):

        # Explanation of the sampling
        self.next_section(skip_animations=False)

        tex_to_color_map = {
            r"\theta": BLUE,
        }

        code = Code(
            code_file="../src/inference_minimal.py",
            background="window",
            language="python",
            add_line_numbers=False,
            formatter_style=Code.get_styles_list()[15],
        )
        code.scale_to_fit_height(0.8 * config.frame_height).to_edge(
            LEFT, buff=0.1
        ).to_edge(UP, buff=0.1)

        self.play(
            Create(code.background),
        )
        self.play(
            Create(code.code_lines[0:12]),
        )

        brace_library = Brace(code.code_lines[0:12], direction=RIGHT, buff=0.6)
        label_library = Tex("Imports and model", font_size=28).next_to(
            brace_library.get_tip(), RIGHT, buff=0.1
        )

        self.play(LaggedStart(GrowFromCenter(brace_library), Write(label_library)))

        self.play(FadeOut(brace_library, label_library), Create(code.code_lines[12:22]))

        brace_model = Brace(code.code_lines[12:22], direction=RIGHT, buff=0.4)
        label_model = Tex("Diffusion constants", font_size=28).next_to(
            brace_model.get_tip(), RIGHT, buff=0.1
        )
        self.play(LaggedStart(GrowFromCenter(brace_model), Write(label_model)))

        self.play(FadeOut(brace_model, label_model), run_time=0.92)

        self.play(Create(code.code_lines[22:47]))
        brace_sampling = Brace(code.code_lines[29:47], direction=RIGHT, buff=0.4)
        label_sampling = Tex("Sampling", font_size=28).next_to(
            brace_sampling.get_tip(), RIGHT, buff=0.1
        )

        self.play(LaggedStart(GrowFromCenter(brace_sampling), Write(label_sampling)))

        xT_circle = Circle(radius=0.5, color=WHITE).move_to(RIGHT * 2)
        xT1_circle = Circle(radius=0.5, color=WHITE).move_to(LEFT * 2)

        xT_label = MathTex(
            r"x_T",
            color=WHITE,
            font_size=32,
        ).move_to(xT_circle)

        xt1_label = MathTex(
            r"x_{T-1}",
            color=WHITE,
            font_size=32,
        ).move_to(xT1_circle)

        p_arrow = DashedVMobject(
            CurvedArrow(
                start_point=xT_circle.get_bottom() + DOWN * 0.05,
                end_point=xT1_circle.get_bottom() + DOWN * 0.05,
                color=BLUE,
                angle=-PI / 2,
            )
        )
        p_label = MathTex(
            r"p_{\theta}(x_{T-1} \mid x_T)",
            color=WHITE,
            font_size=32,
            tex_to_color_map=tex_to_color_map,
        ).next_to(p_arrow, DOWN, buff=0.1)

        diagram = (
            VGroup(
                xT_circle,
                xT1_circle,
                xT_label,
                xt1_label,
                p_arrow,
                p_label,
            )
            .scale_to_fit_width(4)
            .move_to((config.frame_width / 4) * RIGHT)
        )

        self.play(Indicate(code.code_lines[27:28], color=WHITE, scale_factor=1.1))
        self.play(Indicate(code.code_lines[27:28], color=WHITE, scale_factor=1.1))

        self.play(
            LaggedStart(Create(xT_circle), Write(xT_label)),
        )

        self.play(Indicate(code.code_lines[32:33], color=WHITE, scale_factor=1.1))
        self.play(Indicate(code.code_lines[32:33], color=WHITE, scale_factor=1.1))
        self.play(
            LaggedStart(
                LaggedStart(Create(p_arrow), Write(p_label), lag_ratio=0.7),
                LaggedStart(Create(xT1_circle), Write(xt1_label)),
                lag_ratio=0.7,
            ),
            run_time=2,
        )

        posterior_mean = MathTex(
            r"\mu_{\theta}(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \alpha_{t-1}}} \epsilon_\theta(x_t, t) \right)",
            tex_to_color_map=tex_to_color_map,
            font_size=28,
        ).next_to(diagram, UP, buff=0.75)

        self.play(Write(posterior_mean))

        self.play(Indicate(code.code_lines[43:47], color=WHITE, scale_factor=1.1))
        self.play(Indicate(code.code_lines[43:47], color=WHITE, scale_factor=1.1))

        self.play(FadeOut(diagram, brace_sampling, label_sampling, posterior_mean))

        # Complete sampling on MNIST
        self.next_section(skip_animations=False)

        def create_image_with_border(image_path):
            img = (
                ImageMobject(image_path)
                .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
                .scale_to_fit_width(1.5)
            )
            img.add(
                SurroundingRectangle(
                    img, color=WHITE, buff=0.0, stroke_width=2, z_index=1
                )
            )
            return img

        img_list = [
            create_image_with_border(f"../src/img/mnist_samples/sample_{i}_990.png")
            for i in range(8)
        ]

        imgs = (
            Group(
                *img_list,
            )
            .arrange_in_grid(rows=2, cols=4, buff=0.2)
            .move_to(ORIGIN)
            .to_edge(RIGHT, buff=0.2)
        )
        mnist_title = (
            Tex("MNIST samples", font_size=32)
            .next_to(imgs, UP, buff=0.75)
            .set_color(WHITE)
        )
        mnist_title_ul = Underline(mnist_title, color=WHITE)

        self.play(
            LaggedStart(
                LaggedStart(Write(mnist_title), Create(mnist_title_ul), lag_ratio=0.4),
                FadeIn(imgs),
                lag_ratio=0.8,
            ),
            run_time=1.5,
        )

        for t in range(990, -1, -10):
            img_samples_new = []
            for i in range(8):
                img_new = create_image_with_border(
                    f"../src/img/mnist_samples/sample_{i}_{t}.png"
                ).move_to(img_list[i])
                img_samples_new.append(img_new)

            self.add(*img_samples_new)
            self.remove(*img_list)
            img_list = img_samples_new

            self.wait(0.1)

        # Complete sampling on FFHQ
        self.next_section(skip_animations=False)

        new_code = Code(
            code_file="../src/inference_minimal_2.py",
            background="window",
            language="python",
            add_line_numbers=False,
            formatter_style=Code.get_styles_list()[15],
        )
        new_code.scale_to_fit_height(0.8 * config.frame_height).to_edge(
            LEFT, buff=0.1
        ).to_edge(UP, buff=0.1)

        self.play(Uncreate(code.code_lines[7:12], run_time=2))
        self.play(Create(new_code.code_lines[8:12], run_time=2))
        self.play(FadeOut(mnist_title, mnist_title_ul, *img_list))

        img_list = [
            create_image_with_border(f"../src/img/ffhq_samples/sample_{i}_990.png")
            for i in range(8)
        ]
        imgs = (
            Group(
                *img_list,
            )
            .arrange_in_grid(rows=2, cols=4, buff=0.2)
            .move_to(ORIGIN)
            .to_edge(RIGHT, buff=0.2)
        )
        ffhq_title = (
            Tex("FFHQ samples", font_size=32)
            .next_to(imgs, UP, buff=0.75)
            .set_color(WHITE)
        )
        ffhq_title_ul = Underline(ffhq_title, color=WHITE)

        self.play(
            LaggedStart(
                LaggedStart(Write(ffhq_title), Create(ffhq_title_ul), lag_ratio=0.4),
                FadeIn(imgs),
                lag_ratio=0.8,
            ),
            run_time=1.5,
        )

        for t in range(990, -1, -10):
            img_samples_new = []
            for i in range(8):
                img_new = create_image_with_border(
                    f"../src/img/ffhq_samples/sample_{i}_{t}.png"
                ).move_to(img_list[i])
                img_samples_new.append(img_new)
            self.add(*img_samples_new)
            self.remove(*img_list)
            img_list = img_samples_new
            self.wait(0.1)

        vae_img_list = [
            create_image_with_border(f"./img/vae/sample_{i}.png") for i in range(8)
        ]
        vae_imgs = (
            Group(
                *vae_img_list,
            )
            .arrange_in_grid(rows=2, cols=4, buff=0.2)
            .move_to(ORIGIN)
            .to_edge(LEFT, buff=0.2)
        )
        vae_title = (
            Tex("VAE samples", font_size=32)
            .next_to(vae_imgs, UP, buff=0.75)
            .set_color(WHITE)
        )
        vae_title_ul = Underline(vae_title, color=WHITE)

        self.play(
            FadeOut(
                new_code.code_lines[8:12],
                code.code_lines[0:7],
                code.code_lines[12:],
                code.background,
            )
        )
        self.play(
            LaggedStart(
                LaggedStart(Write(vae_title), Create(vae_title_ul), lag_ratio=0.4),
                FadeIn(vae_imgs),
                lag_ratio=0.8,
            ),
            run_time=1.5,
        )

        self.play(FadeOut(*self.mobjects, shift=0.5 * DOWN))

        self.wait(1)


if __name__ == "__main__":
    scene = Scene2_9()
    scene.render()
