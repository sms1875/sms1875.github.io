require 'yaml'

def generate_post(problem_dir)
  readme_path = File.join(problem_dir, 'README.md')
  return unless File.exist?(readme_path)

  content = File.read(readme_path)
  metadata = content.split("\n\n").first
  description = content.split("\n\n")[1..-1].join("\n\n")

  code_file = Dir.glob(File.join(problem_dir, '*')).find { |f| f != readme_path }
  language = File.extname(code_file)[1..-1]

  platform = problem_dir.split('/')[1] # Assumes directory structure like coding_test_repository/baekjoon/...

  filename = "#{Time.now.strftime('%Y-%m-%d')}-#{File.basename(problem_dir)}.md"

  post_content = <<~CONTENT
    ---
    title: "#{metadata.split("\n").first}"
    categories: [Coding Test, #{platform.capitalize}]
    tags: [coding test, #{platform}]
    language: #{language}
    ---

    #{description}
  CONTENT

  File.write(File.join('_coding_tests', filename), post_content)
end

Dir.glob('coding_test_repository/**/*').select { |f| File.directory?(f) }.each do |dir|
  generate_post(dir)
end
